# Data Contract

`collect_teacher_rollouts.py` appends one raw JSON object for every `(task_id, attempt_index)` pair. A raw trajectory always has a unique `trajectory_id` and preserves `messages`, executed `steps`, `initial_result`, `terminal_result`, `status`, `done`, `final_reward`, and any `error`.

`build_sft_data.py` deterministically derives four files from the complete raw JSONL:

- `accepted.jsonl` contains the original raw trajectories that pass all checks.
- `rejected.jsonl` contains `trajectory_id`, `task_id`, `status`, and structured `reject_reasons`.
- `stats.json` contains total, accepted, rejected, and rejection counts.
- `sft_openai_messages.jsonl` contains only `trajectory_id`, `task_id`, sanitized `messages`, and shared `tools`.

An accepted trajectory must finish without an error, perform a purchase, receive an environment terminal result, and have every `reward_detail` component (`r_type`, `r_att`, `r_option`, `r_price`) equal to 1. Tool call JSON and its mapped environment action must also agree.

SFT rows intentionally exclude `goal`, standard answers, `purchase`, `reward_detail`, terminal reward, and all other environment-only fields. They can be passed to Hugging Face `apply_chat_template(..., tools=row["tools"])`.
