"""Validate generated-token loss masks for the enhanced adapter."""


def verify_mask(response_ids, response_mask):
    if len(response_ids) != len(response_mask):
        raise ValueError("Response tokens and loss mask differ in length")
    if not set(response_mask) <= {0, 1}:
        raise ValueError("Loss mask must be binary")
    if not any(response_mask):
        raise ValueError("No generated action tokens available for learning")
    return {
        "generated_tokens": sum(response_mask),
        "non_action_tokens": len(response_mask) - sum(response_mask),
    }
