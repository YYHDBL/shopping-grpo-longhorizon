from __future__ import annotations

import math
import time

import hydra
import torch

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


def _local_agentic_stats(input_ids, loss_mask):
    lengths = input_ids.offsets().diff().to(dtype=torch.float64)
    masks = loss_mask.unbind()
    supervised = torch.tensor(
        [mask.sum().item() for mask in masks], dtype=torch.float64
    )
    assistant_turns = sum(
        int(mask[0].item() == 1)
        + int(((mask[1:] == 1) & (mask[:-1] == 0)).sum().item())
        for mask in masks
    )
    return {
        "sequence_count": len(masks),
        "total_tokens": lengths.sum().item(),
        "supervised_tokens": supervised.sum().item(),
        "supervised_sequences": (supervised > 0).sum().item(),
        "assistant_turns": assistant_turns,
        "min_sequence_length": lengths.min().item(),
        "max_sequence_length": lengths.max().item(),
    }


class AgenticSFTTrainer(SFTTrainer):
    def _build_engine(self):
        super()._build_engine()
        train_batch = self.training_client.train_batch

        def monitored_train_batch(data):
            started_at = time.perf_counter()
            local = _local_agentic_stats(data["input_ids"], data["loss_mask"])
            device = torch.device(self.config.trainer.device, torch.cuda.current_device())
            totals = torch.tensor(
                [
                    local["sequence_count"],
                    local["total_tokens"],
                    local["supervised_tokens"],
                    local["supervised_sequences"],
                    local["assistant_turns"],
                ],
                dtype=torch.float64,
                device=device,
            )
            minimum = torch.tensor(local["min_sequence_length"], dtype=torch.float64, device=device)
            maximum = torch.tensor(local["max_sequence_length"], dtype=torch.float64, device=device)
            dp_group = self.engine.get_data_parallel_group()
            if dp_group is not None:
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM, group=dp_group)
                torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=dp_group)
                torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=dp_group)

            output = train_batch(data=data)
            elapsed = torch.tensor(time.perf_counter() - started_at, dtype=torch.float64, device=device)
            if dp_group is not None:
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX, group=dp_group)

            if output is not None:
                metrics = tu.get(output, "metrics")
                sequences, tokens, supervised, supervised_sequences, turns = totals.tolist()
                seconds = elapsed.item()
                context = tokens - supervised
                metrics.update(
                    {
                        "train/perplexity": math.exp(float(metrics["loss"])),
                        "train/sequence_count": sequences,
                        "train/assistant_turn_count": turns,
                        "train/assistant_turns_per_sequence": turns / sequences,
                        "train/supervised_sequence_ratio": supervised_sequences / sequences,
                        "train/sequence_length_mean": tokens / sequences,
                        "train/sequence_length_min": minimum.item(),
                        "train/sequence_length_max": maximum.item(),
                        "train/supervised_tokens": supervised,
                        "train/context_tokens": context,
                        "train/supervised_tokens_per_sequence": supervised / sequences,
                        "train/context_tokens_per_sequence": context / sequences,
                        "train/supervised_token_ratio": supervised / tokens,
                        "perf/step_time_seconds": seconds,
                        "perf/tokens_per_second": tokens / seconds,
                        "perf/supervised_tokens_per_second": supervised / seconds,
                    }
                )
            return output

        self.training_client.train_batch = monitored_train_batch


def run_sft(config):
    initialize_global_process_group()
    trainer = AgenticSFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="pkg://verl.trainer.config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    auto_set_device(config)
    run_sft(config)


if __name__ == "__main__":
    main()
