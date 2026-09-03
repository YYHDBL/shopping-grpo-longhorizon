"""TRACE credit assignment for multi-turn Shopping GRPO rollouts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence


def canonical_purchase_target(asin: object, options: object) -> str:
    """Build the private, deterministic target scored by the frozen reference."""
    if not isinstance(asin, str) or not asin.strip():
        raise ValueError("TRACE target asin must be a non-empty string")
    if options is None:
        options = []
    if not isinstance(options, (Mapping, list, tuple)):
        raise ValueError("TRACE target options must be an object or array")
    payload = {"asin": asin.strip(), "options": options}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("TRACE target must be JSON serializable") from exc
    return f"最终应购买商品：{encoded}"


def trace_turn_layout(
    response_mask: Sequence[int], response_attention: Sequence[int]
) -> list[tuple[tuple[int, int], int]]:
    """Return ``((assistant_start, assistant_end), post-tool-prefix-end)``."""
    if len(response_mask) != len(response_attention):
        raise ValueError("response mask and attention must have equal length")
    if any(value not in (0, 1, False, True) for value in response_mask):
        raise ValueError("response mask must be binary")
    if any(value not in (0, 1, False, True) for value in response_attention):
        raise ValueError("response attention must be binary")
    try:
        active_end = response_attention.index(0)
    except ValueError:
        active_end = len(response_attention)
    if any(response_attention[active_end:]):
        raise ValueError("response attention must be right padded")

    layout = []
    index = 0
    while index < active_end:
        while index < active_end and not response_mask[index]:
            index += 1
        assistant_start = index
        while index < active_end and response_mask[index]:
            index += 1
        assistant_end = index
        while index < active_end and not response_mask[index]:
            index += 1
        if assistant_start < assistant_end < index:
            layout.append(((assistant_start, assistant_end), index))
    return layout


def turn_rewards(
    mean_target_log_probs: Sequence[float],
    *,
    outcome_advantage: float,
    epsilon: float,
    horizon: int,
    discount: float,
    terminal_weight: float,
) -> list[float]:
    """Compute TRACE log-ratio credit, K-step backup and terminal fill."""
    if len(mean_target_log_probs) < 1:
        raise ValueError("TRACE requires at least the initial prefix score")
    if epsilon <= 0 or horizon <= 0 or not 0 <= discount <= 1 or terminal_weight < 0:
        raise ValueError("invalid TRACE hyperparameters")
    gaps = [-float(score) + epsilon for score in mean_target_log_probs]
    if any(not math.isfinite(gap) or gap <= 0 for gap in gaps):
        raise ValueError("TRACE target log-probabilities produce an invalid remaining gap")
    deltas = [
        math.log(left / right)
        for left, right in zip(gaps[:-1], gaps[1:], strict=True)
    ]
    rewards = []
    turn_count = len(deltas)
    for turn in range(turn_count):
        stop = min(turn + horizon, turn_count)
        weights = [discount**offset for offset in range(stop - turn)]
        credit = sum(
            weight * deltas[index]
            for weight, index in zip(weights, range(turn, stop), strict=True)
        ) / sum(weights)
        if stop == turn_count:
            credit += terminal_weight * discount ** (turn_count - turn) * float(
                outcome_advantage
            )
        rewards.append(credit)
    return rewards


def mixed_token_advantages(
    *,
    outcome_advantage: float,
    response_mask: Sequence[int],
    response_attention: Sequence[int],
    mean_target_log_probs: Sequence[float],
    epsilon: float,
    horizon: int,
    discount: float,
    terminal_weight: float,
    outcome_weight: float,
    turn_weight: float,
) -> list[float]:
    """Map turn rewards onto assistant tokens while keeping tool tokens masked."""
    layout = trace_turn_layout(response_mask, response_attention)
    if len(mean_target_log_probs) != len(layout) + 1:
        raise ValueError("TRACE prefix scores do not match tool-turn boundaries")
    result = [
        outcome_weight * float(outcome_advantage)
        if response_mask[index] and response_attention[index]
        else 0.0
        for index in range(len(response_mask))
    ]
    rewards = turn_rewards(
        mean_target_log_probs,
        outcome_advantage=outcome_advantage,
        epsilon=epsilon,
        horizon=horizon,
        discount=discount,
        terminal_weight=terminal_weight,
    )
    for ((start, end), _), reward in zip(layout, rewards, strict=True):
        result[start:end] = [
            outcome_weight * float(outcome_advantage) + turn_weight * reward
        ] * (end - start)
    return result


def build_trace_score_batch(batch, tokenizer, *, max_sequence_length: int):
    """Pack all rollout prefixes for one batched frozen-reference forward pass."""
    import torch
    from verl import DataProto
    from verl.utils.torch_functional import compute_position_id_with_mask

    responses = batch.batch["responses"]
    response_length = responses.shape[1]
    attention = batch.batch["attention_mask"]
    response_attention = attention[:, -response_length:]
    prompt_attention = attention[:, :-response_length]
    targets = batch.non_tensor_batch.get("trace_target")
    if targets is None or len(targets) != len(responses):
        raise ValueError("TRACE requires one private target per rollout")

    prefixes: list[list[int]] = []
    target_tokens: list[list[int]] = []
    state_counts: list[int] = []
    for row in range(len(responses)):
        layout = trace_turn_layout(
            batch.batch["response_mask"][row].tolist(),
            response_attention[row].tolist(),
        )
        boundaries = [0, *[prefix_end for _, prefix_end in layout]]
        prompt = batch.batch["prompts"][row][prompt_attention[row].bool()].tolist()
        response = responses[row].tolist()
        target = tokenizer.encode(str(targets[row]), add_special_tokens=False)
        if not target or len(prompt) + len(target) > max_sequence_length:
            raise ValueError("TRACE target does not fit the configured reference context")
        response_budget = max_sequence_length - len(prompt) - len(target)
        for boundary in boundaries:
            visible_response = response[:boundary]
            # ponytail: preserve the task prompt and recent state; use structured
            # context compaction if scorer truncation becomes a measured bottleneck.
            visible_response = visible_response[-response_budget:] if response_budget else []
            prefixes.append([*prompt, *visible_response])
            target_tokens.append(list(target))
        state_counts.append(len(boundaries))

    pad_id = int(tokenizer.pad_token_id)
    max_prefix = max(map(len, prefixes))
    max_target = max(map(len, target_tokens))
    input_ids = []
    masks = []
    padded_prefixes = []
    padded_targets = []
    target_masks = []
    for prefix, target in zip(prefixes, target_tokens, strict=True):
        left_pad = [pad_id] * (max_prefix - len(prefix))
        right_pad = [pad_id] * (max_target - len(target))
        padded_prefix = [*left_pad, *prefix]
        padded_target = [*target, *right_pad]
        input_ids.append([*padded_prefix, *padded_target])
        masks.append(
            [0] * len(left_pad)
            + [1] * len(prefix)
            + [1] * len(target)
            + [0] * len(right_pad)
        )
        padded_prefixes.append(padded_prefix)
        padded_targets.append(padded_target)
        target_masks.append([1] * len(target) + [0] * len(right_pad))

    device = responses.device
    tensor = lambda values: torch.tensor(values, dtype=torch.long, device=device)
    attention_mask = tensor(masks)
    score_batch = DataProto.from_dict(
        tensors={
            "prompts": tensor(padded_prefixes),
            "responses": tensor(padded_targets),
            "input_ids": tensor(input_ids),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "response_mask": tensor(target_masks),
        }
    )
    return score_batch, state_counts


def apply_trace_advantages(
    batch,
    flat_mean_target_log_probs,
    state_counts: Sequence[int],
    **parameters: float | int,
) -> dict[str, float]:
    """Replace outcome-only GRPO advantages with TRACE mixed advantages."""
    import torch

    if sum(state_counts) != len(flat_mean_target_log_probs):
        raise ValueError("TRACE scorer output count does not match rollout states")
    response_length = batch.batch["responses"].shape[1]
    response_attention = batch.batch["attention_mask"][:, -response_length:]
    mixed_rows = []
    all_rewards = []
    offset = 0
    for row, state_count in enumerate(state_counts):
        mask = batch.batch["response_mask"][row]
        active = torch.nonzero(mask, as_tuple=False)
        outcome = float(batch.batch["advantages"][row, active[0, 0]]) if len(active) else 0.0
        scores = flat_mean_target_log_probs[offset : offset + state_count].tolist()
        offset += state_count
        layout = trace_turn_layout(mask.tolist(), response_attention[row].tolist())
        rewards = turn_rewards(
            scores,
            outcome_advantage=outcome,
            epsilon=float(parameters["epsilon"]),
            horizon=int(parameters["horizon"]),
            discount=float(parameters["discount"]),
            terminal_weight=float(parameters["terminal_weight"]),
        )
        all_rewards.extend(rewards)
        mixed_rows.append(
            mixed_token_advantages(
                outcome_advantage=outcome,
                response_mask=mask.tolist(),
                response_attention=response_attention[row].tolist(),
                mean_target_log_probs=scores,
                epsilon=float(parameters["epsilon"]),
                horizon=int(parameters["horizon"]),
                discount=float(parameters["discount"]),
                terminal_weight=float(parameters["terminal_weight"]),
                outcome_weight=float(parameters["outcome_weight"]),
                turn_weight=float(parameters["turn_weight"]),
            )
        )
        if len(layout) != state_count - 1:
            raise ValueError("TRACE layout changed while applying advantages")
    mixed = torch.tensor(
        mixed_rows,
        dtype=batch.batch["advantages"].dtype,
        device=batch.batch["advantages"].device,
    )
    batch.batch["advantages"] = mixed
    batch.batch["returns"] = mixed
    if not all_rewards:
        return {"trace/turns": 0.0, "trace/turn_reward_mean": 0.0}
    return {
        "trace/turns": float(len(all_rewards)),
        "trace/turn_reward_mean": float(sum(all_rewards) / len(all_rewards)),
        "trace/positive_turn_ratio": float(
            sum(reward > 0 for reward in all_rewards) / len(all_rewards)
        ),
    }
