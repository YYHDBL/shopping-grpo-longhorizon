import os
from pathlib import Path

import ray


@ray.remote(num_gpus=1, num_cpus=1)
class FSDPProbe:
    def identity(self):
        import torch

        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        assigned = ray.get_runtime_context().get_accelerator_ids()["GPU"]
        assert len(assigned) == 1, assigned
        return str(properties.uuid), properties.pci_bus_id, assigned

    def master_address(self):
        from verl.single_controller.base.worker import WorkerHelper

        return WorkerHelper().get_available_master_addr_port()

    def check(self, rank, world_size, address, port):
        import copy

        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from verl.single_controller.base import Worker
        from verl.utils.distributed import initialize_global_process_group_ray

        os.environ.update(
            RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK="0",
            MASTER_ADDR=address, MASTER_PORT=str(port),
        )
        Worker()
        initialize_global_process_group_ray(timeout_second=60)

        # 使用确定性小模型验证真实 FSDP 参数同步、前向、反向和优化器更新。
        torch.manual_seed(42)
        module = torch.nn.Linear(8, 4)
        reference = copy.deepcopy(module)
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        reference(inputs).square().mean().backward()
        reference_optimizer.step()

        model = FSDP(module, device_id=0, sync_module_states=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = model(inputs.cuda()).square().mean()
        loss.backward()
        optimizer.step()
        with FSDP.summon_full_params(model):
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual.cpu(), expected)
        dist.destroy_process_group()
        return {"rank": rank, "loss": loss.item(), "fsdp": "PASS"}


def main():
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(visible) >= 2 and all(item.startswith("GPU-") for item in visible), visible
    root = Path(__file__).resolve().parents[1]
    ray.init(
        address="local",
        num_gpus=len(visible),
        num_cpus=len(visible),
        include_dashboard=False,
        object_store_memory=128 * 1024 * 1024,
        _temp_dir=str(root / "logs"),
        runtime_env={
            "env_vars": {"WG_BACKEND": "ray"},
            "worker_process_setup_hook": "shopping_grpo.training.grpo.compat.install_torch_padding_fallback"
        },
    )
    try:
        workers = [FSDPProbe.remote() for _ in visible]
        identities = ray.get([worker.identity.remote() for worker in workers], timeout=120)
        for uuid, bus, assigned in identities:
            assert assigned == ["GPU-" + uuid], (uuid, assigned)
            print({"uuid": uuid, "pci_bus": hex(bus), "assigned": assigned}, flush=True)
        assert len({uuid for uuid, _, _ in identities}) == len(visible), identities
        address, port = ray.get(workers[0].master_address.remote(), timeout=30)
        results = ray.get(
            [worker.check.remote(rank, len(visible), address, port) for rank, worker in enumerate(workers)],
            timeout=120,
        )
        for result in results:
            print(result, flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
