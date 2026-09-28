import torch

from shopping_grpo.training.sft.trainer import _local_agentic_stats


def test_local_agentic_stats_counts_masked_turns():
    input_ids = torch.nested.nested_tensor(
        [torch.arange(5), torch.arange(4)], layout=torch.jagged
    )
    loss_mask = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 1, 0, 1]), torch.tensor([1, 1, 0, 0])],
        layout=torch.jagged,
    )

    assert _local_agentic_stats(input_ids, loss_mask) == {
        "sequence_count": 2,
        "total_tokens": 9.0,
        "supervised_tokens": 5.0,
        "supervised_sequences": 2,
        "assistant_turns": 3,
        "min_sequence_length": 4.0,
        "max_sequence_length": 5.0,
    }
