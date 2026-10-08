"""Instrument the existing veRL ShoppingToolAgentLoop, without a new trainer."""

import json
import time
from pathlib import Path
from uuid import uuid4

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.enhanced_protocol import (
    verify_mask,
)
from shopping_grpo.training.grpo.adapter.legacy_reward import select_legacy_reward
from shopping_grpo.training.grpo.adapter.runtime import current_runtime_state


class RecordingEnv(ShopAgentEnv):
    def step(self, action):
        result = super().step(action)
        state = current_runtime_state.get()
        if state is not None:
            # Public observations and terminal results are audit-only, not model prompts.
            state.setdefault("pilot_environment_events", []).append(
                {"action": action, "result": result}
            )
        return result


class EnhancedShoppingAgentLoop(ShoppingToolAgentLoop):
    def __init__(
        self,
        *args,
        select_original_reward=False,
        trajectory_directory=None,
        system_prompt_file=None,
        **kwargs,
    ):
        self.system_prompt = Path(system_prompt_file).read_text() if system_prompt_file else None
        self.select_original_reward = select_original_reward
        self.trajectory_directory = trajectory_directory
        super().__init__(*args, env_factory=RecordingEnv, **kwargs)

    async def _handle_pending_state(self, agent_data, sampling_params):
        state = current_runtime_state.get()
        state["task_instruction"] = agent_data.messages[-1]["content"]
        agent_data.messages = [dict(m) for m in agent_data.messages]
        if self.system_prompt is not None:
            for message in agent_data.messages:
                if message.get("role") == "system":
                    message["content"] = self.system_prompt
                    break
            else:
                agent_data.messages.insert(0, {"role": "system", "content": self.system_prompt})
        agent_data.messages[-1]["content"] += "\n\n" + state["latest_observation"]
        agent_data.extra_fields["pilot_initial_messages"] = agent_data.messages
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        # Parent compacts before generating. Save the resulting authoritative
        # token stream below, along with per-turn generated tokens.
        sampling_params = dict(sampling_params)
        turns_before = agent_data.assistant_turns
        result = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination
        )
        if agent_data.assistant_turns == turns_before:
            return result
        agent_data.extra_fields.setdefault("pilot_generation_events", []).append(
            {
                "prompt_and_response_ids": list(agent_data.prompt_ids),
                "response_mask": list(agent_data.response_mask),
                "generated_response_ids": list(agent_data.response_ids),
                "policy_metadata": {
                    "max_global_steps": agent_data.extra_fields.get("max_global_steps")
                },
            }
        )
        return result

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        state = current_runtime_state.get()
        agent_data.extra_fields["pilot_environment_events"] = state.get(
            "pilot_environment_events", []
        )
        agent_data.extra_fields["frozen_support_events"] = state.get("frozen_support_events", [])
        return result

    async def _handle_processing_tools_state(self, agent_data):
        before = len(agent_data.response_mask)
        result = await super()._handle_processing_tools_state(agent_data)
        added = agent_data.response_mask[before:]
        if any(added):
            raise RuntimeError("Environment tool tokens entered the policy loss mask")
        return result

    async def run(self, sampling_params, **kwargs):
        # The parent owns session lifetime; capture state before its close via
        # generation/tool callbacks, never keep a shared mutable rollout state.
        output = await super().run(sampling_params, **kwargs)
        info = output.extra_fields["shopping"]
        if self.select_original_reward:
            select_legacy_reward(output, output.extra_fields.get("pilot_environment_events", []))
        mask = verify_mask(output.response_ids, output.response_mask)
        info["loss_mask_audit"] = mask
        if self.trajectory_directory is None:
            return output
        directory = Path(self.trajectory_directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "task_id": info["task_id"],
            "sampling_params": sampling_params,
            "captured_at": time.time(),
            "raw_prompt": kwargs.get("raw_prompt"),
            "prompt_ids": output.prompt_ids,
            "response_ids": output.response_ids,
            "response_mask": output.response_mask,
            "response_logprobs": output.response_logprobs,
            "decoded_response": self.tokenizer.decode(output.response_ids),
            "reward_score": output.reward_score,
            "extra_fields": {k: v for k, v in output.extra_fields.items() if k != "trace_target"},
        }
        (directory / (str(info["task_id"]) + "-" + uuid4().hex + ".json")).write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else x.item(),
            )
            + "\n"
        )
        return output
