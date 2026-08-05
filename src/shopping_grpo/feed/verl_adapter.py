"""veRL 0.8 adapter for the long-horizon Feed shopping environment.

The adapter has two deliberately separate channels:

* tool responses contain only the public observation and public tool result;
* evaluator state (episode return, event diagnostics, and process-credit vectors)
  is attached to ``AgentLoopOutput.extra_fields`` after the episode terminates.

veRL 0.8 converts ``AgentLoopOutput.reward_score`` into a single reward at the
end of the generated sequence.  The process-credit vectors built here are
therefore metadata only.  They are useful for audits and a future patched
advantage path, but this module does **not** claim that vanilla veRL consumes
them when computing GRPO advantages.

All veRL imports have dependency-free fallbacks.  This keeps dataset generation,
unit tests, and the static preflight usable on CPU-only development machines.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from uuid import uuid4

from shopping_grpo.feed.actions import action_reject_reason
from shopping_grpo.feed.catalog import ProductCatalog
from shopping_grpo.feed.credit import event_credit_by_source
from shopping_grpo.feed.evidence import has_product_evidence
from shopping_grpo.feed.observation import FEED_OBSERVATION_VERSION
from shopping_grpo.feed.schema import EpisodeSeed
from shopping_grpo.feed.simulator import (
    FEED_ENVIRONMENT_VERSION,
    FEED_REWARD_VERSION,
    FeedActionGuardError,
    FeedShoppingEnv,
)
from shopping_grpo.feed.tools import (
    COMMIT_TOOL_NAME,
    FEED_TOOL_NAMES,
    INFO_TOOL_NAMES,
    MAX_INFO_TOOL_CALLS_PER_VIDEO,
    validate_tool_arguments,
)


FEED_TOOLS_VERSION = "feed-tools-v1"
FEED_TOOL_PAYLOAD_VERSION = "feed-tool-delta-v1"
CREDIT_MODES = frozenset({"terminal", "rtg", "event", "counterfactual"})
PROCESS_CREDIT_CONTRACT = "feed-process-credit-v1"
NATIVE_REWARD_BOUNDARY = "scalar_terminal_reward_only"
SHOPPING_PATCH_COMPATIBILITY = "feed-shopping-extra-v1"


try:  # pragma: no cover - exercised in the pinned GPU runtime.
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
    from verl.utils.rollout_trace import rollout_trace_op

    VERL_AVAILABLE = True
except ImportError:  # dependency-free development fallback
    VERL_AVAILABLE = False

    class ToolResponse:
        """Small subset of veRL's pydantic response used by local tests."""

        def __init__(self, text=None, image=None, video=None):
            self.text = text
            self.image = image
            self.video = video

    class BaseTool:
        """Import-compatible fallback for :class:`verl.tools.BaseTool`."""

        def __init__(self, config=None, tool_schema=None):
            self.config = config or {}
            self.tool_schema = tool_schema
            function = (
                tool_schema.get("function", {})
                if isinstance(tool_schema, Mapping)
                else getattr(tool_schema, "function", None)
            )
            self.name = (
                function.get("name")
                if isinstance(function, Mapping)
                else getattr(function, "name", None)
            )

    class AgentState(str, Enum):
        PENDING = "pending"
        GENERATING = "generating"
        PROCESSING_TOOLS = "processing_tools"
        TERMINATED = "terminated"

    class ToolAgentLoop:
        """Fallback that can be monkey-patched in tests but cannot generate."""

        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def run(self, sampling_params, **kwargs):
            del sampling_params, kwargs
            raise RuntimeError(
                "veRL is not installed; FeedToolAgentLoop generation is unavailable"
            )

        async def _handle_processing_tools_state(self, agent_data):
            del agent_data
            return AgentState.TERMINATED

    def rollout_trace_op(function):
        return function


class FeedPublicPayloadError(ValueError):
    """A tool attempted to expose evaluator-only or latent state."""


@dataclass(frozen=True)
class FeedTask:
    """One resolved training task: fixed episode seed plus product truth."""

    episode_seed: EpisodeSeed
    catalog: ProductCatalog
    calibration: Any = None


class FeedTaskStore:
    """Thread-safe in-memory task registry for AgentLoop workers.

    A worker may also receive a serialized ``episode_seed`` and catalog in the
    sample kwargs.  The store is useful when those larger immutable objects are
    loaded once per worker instead of repeated in every parquet row.
    """

    def __init__(self, tasks: Mapping[Any, Any] | None = None):
        self._tasks: dict[str, FeedTask] = {}
        self._lock = RLock()
        for key, value in (tasks or {}).items():
            self.register(key, value)

    def register(self, key: Any, task: Any) -> None:
        normalized = _coerce_task(task)
        with self._lock:
            self._tasks[str(key)] = normalized
            self._tasks.setdefault(normalized.episode_seed.episode_id, normalized)

    def resolve(self, key: Any) -> FeedTask:
        with self._lock:
            try:
                return self._tasks[str(key)]
            except KeyError as exc:
                raise KeyError(f"unknown Feed task: {key!r}") from exc

    def __len__(self) -> int:
        with self._lock:
            return len({task.episode_seed.episode_id for task in self._tasks.values()})


current_feed_environment: ContextVar = ContextVar(
    "feed_shopping_environment", default=None
)
current_feed_runtime_state: ContextVar = ContextVar(
    "feed_shopping_runtime_state", default=None
)


def make_feed_runtime_state(episode_id: str) -> dict[str, Any]:
    """Create trajectory-local bookkeeping without copying latent simulator state."""

    return {
        "episode_id": str(episode_id),
        "latest_observation": None,
        "tool_calls": [],
        "commits": [],
        "assistant_commit_positions": [],
        "pending_commit_position": None,
        "done": False,
        "terminate": False,
        "infrastructure_invalid": False,
        "termination_reason": None,
        "error_type": None,
        "guard_rejections": 0,
    }


class FeedTrajectorySession:
    """Bind one Feed environment to the current coroutine and always unbind it."""

    def __init__(
        self,
        task: FeedTask,
        *,
        max_info_calls_per_video: int = MAX_INFO_TOOL_CALLS_PER_VIDEO,
        env_factory=FeedShoppingEnv,
    ) -> None:
        self.task = task
        self.max_info_calls_per_video = int(max_info_calls_per_video)
        self.env_factory = env_factory
        self.env = None
        self.state = None
        self._environment_token = None
        self._state_token = None

    async def start(self) -> dict[str, Any]:
        if self.env is not None:
            raise RuntimeError("Feed trajectory session has already started")
        self.env = self.env_factory(
            self.task.episode_seed,
            self.task.catalog,
            calibration=self.task.calibration,
            max_info_calls_per_video=self.max_info_calls_per_video,
        )
        observation = self.env.observation()
        _validate_public_tool_payload(observation)
        self.state = make_feed_runtime_state(self.task.episode_seed.episode_id)
        self.state["latest_observation"] = observation
        self._environment_token = current_feed_environment.set(self.env)
        self._state_token = current_feed_runtime_state.set(self.state)
        return self.state

    async def close(self) -> None:
        if self._state_token is not None:
            current_feed_runtime_state.reset(self._state_token)
        if self._environment_token is not None:
            current_feed_environment.reset(self._environment_token)
        self.env = None
        self.state = None
        self._state_token = None
        self._environment_token = None


class FeedShoppingTool(BaseTool):
    """Execute the canonical seven information tools plus one commit tool.

    Every invocation uses the coroutine-local environment installed by
    :class:`FeedTrajectorySession`.  The return reward is always ``0.0``;
    episode reward is settled only by :class:`FeedToolAgentLoop` after terminal.
    """

    async def create(self, instance_id=None, **kwargs):
        del kwargs
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ):
        del instance_id, kwargs
        env = current_feed_environment.get()
        state = current_feed_runtime_state.get()
        if env is None or state is None:
            raise RuntimeError(
                "FeedShoppingTool executed without a trajectory-local session"
            )
        if state["done"] or state["terminate"]:
            return _public_error("environment_terminal"), 0.0, {
                "accepted": False,
                "done": True,
            }
        if self.name not in FEED_TOOL_NAMES:
            return _public_error("unknown_tool"), 0.0, {
                "accepted": False,
                "done": False,
            }

        try:
            normalized = validate_tool_arguments(self.name, parameters or {})
        except Exception as exc:
            state["guard_rejections"] += 1
            return _public_error("invalid_arguments", exc.__class__.__name__), 0.0, {
                "accepted": False,
                "done": False,
            }

        reason = action_reject_reason(
            self.name,
            normalized,
            state["latest_observation"],
        )
        if reason is not None:
            state["guard_rejections"] += 1
            code, _, detail = str(reason).partition(":")
            return _public_error(code, detail or None), 0.0, {
                "accepted": False,
                "done": False,
            }

        feed_step = int(getattr(env, "step_index", 0))
        try:
            raw_result = await asyncio.to_thread(env.call_tool, self.name, normalized)
            observation = env.observation()
            payload = build_public_tool_payload(
                self.name,
                raw_result,
                observation,
            )
            public_payload = _validate_public_tool_payload(payload)
        except FeedActionGuardError as exc:
            # A simulator-side guard is still a model-action rejection, not an
            # infrastructure outage.  Do not expose its message, which may gain
            # evaluator detail in future environment versions.
            state["guard_rejections"] += 1
            return _public_error("runtime_guard_rejection", exc.__class__.__name__), 0.0, {
                "accepted": False,
                "done": False,
            }
        except Exception as exc:
            # Do not echo an exception message: catalog rows or simulator details may
            # contain evaluator-only material.  The exception class is enough for ops.
            state["terminate"] = True
            state["infrastructure_invalid"] = True
            state["termination_reason"] = "tool_execution_failed"
            state["error_type"] = exc.__class__.__name__
            return _public_error("tool_execution_failed", exc.__class__.__name__), 0.0, {
                "accepted": False,
                "done": False,
            }

        done = bool(observation.get("done"))
        record = {
            "tool": self.name,
            "feed_step": feed_step,
            "done": done,
        }
        state["tool_calls"].append(record)
        state["latest_observation"] = observation
        if self.name == COMMIT_TOOL_NAME:
            position = state.get("pending_commit_position")
            state["commits"].append({"feed_step": feed_step, "done": done})
            state["assistant_commit_positions"].append(
                int(position) if isinstance(position, int) else None
            )
        if done:
            state["done"] = True
            state["terminate"] = True
            state["termination_reason"] = "environment_done"
        return ToolResponse(text=_canonical_json(public_payload)), 0.0, {
            "accepted": True,
            "done": done,
            "feed_step": feed_step,
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        del instance_id, kwargs
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # veRL releases a tool instance after each call.  The AgentLoop owns the
        # longer-lived environment session and closes it in ``finally``.
        del instance_id, kwargs


class FeedToolAgentLoop(ToolAgentLoop):
    """Bind a fixed Feed task to veRL's native multi-turn ToolAgentLoop."""

    def __init__(
        self,
        *args,
        credit_mode: str = "terminal",
        credit_gamma: float = 0.99,
        min_feed_steps: int = 24,
        max_feed_steps: int = 48,
        max_info_calls_per_video: int = MAX_INFO_TOOL_CALLS_PER_VIDEO,
        expected_min_turn_capacity: int = 128,
        context_window_tokens: int = 65536,
        tool_payload_version: str = FEED_TOOL_PAYLOAD_VERSION,
        process_credit_metadata_only: bool = True,
        native_advantage_integration: bool = False,
        required_environment_version: str = FEED_ENVIRONMENT_VERSION,
        required_observation_version: str = FEED_OBSERVATION_VERSION,
        required_reward_version: str = FEED_REWARD_VERSION,
        required_tools_version: str = FEED_TOOLS_VERSION,
        task_store: FeedTaskStore | Mapping[Any, Any] | None = None,
        catalog: Any = None,
        catalog_path: str | Path | None = None,
        calibration: Any = None,
        env_factory=FeedShoppingEnv,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if credit_mode not in CREDIT_MODES:
            raise ValueError(
                f"credit_mode must be one of {sorted(CREDIT_MODES)}, got {credit_mode!r}"
            )
        if not math.isfinite(float(credit_gamma)) or not 0.0 <= float(credit_gamma) <= 1.0:
            raise ValueError("credit_gamma must be finite and between zero and one")
        if int(min_feed_steps) < 1 or int(max_feed_steps) < int(min_feed_steps):
            raise ValueError("Feed step bounds must be positive and ordered")
        if not 0 <= int(max_info_calls_per_video) <= MAX_INFO_TOOL_CALLS_PER_VIDEO:
            raise ValueError(
                "max_info_calls_per_video must be between zero and "
                f"{MAX_INFO_TOOL_CALLS_PER_VIDEO}"
            )
        if int(expected_min_turn_capacity) < 128:
            raise ValueError("expected_min_turn_capacity must be at least 128")
        if int(context_window_tokens) < 8192:
            raise ValueError("context_window_tokens must be at least 8192")
        if str(tool_payload_version) != FEED_TOOL_PAYLOAD_VERSION:
            raise ValueError(
                "Feed tool payload version mismatch: "
                f"expected {FEED_TOOL_PAYLOAD_VERSION!r}, got {tool_payload_version!r}"
            )
        if not bool(process_credit_metadata_only):
            raise ValueError("veRL 0.8 Feed process credit must remain metadata-only")
        if bool(native_advantage_integration):
            raise ValueError(
                "native_advantage_integration is unavailable in vanilla veRL 0.8"
            )
        self.credit_mode = str(credit_mode)
        self.credit_gamma = float(credit_gamma)
        self.min_feed_steps = int(min_feed_steps)
        self.max_feed_steps = int(max_feed_steps)
        self.max_info_calls_per_video = int(max_info_calls_per_video)
        self.expected_min_turn_capacity = int(expected_min_turn_capacity)
        self.context_window_tokens = int(context_window_tokens)
        self.tool_payload_version = str(tool_payload_version)
        self.required_environment_version = str(required_environment_version)
        self.required_observation_version = str(required_observation_version)
        self.required_reward_version = str(required_reward_version)
        self.required_tools_version = str(required_tools_version)
        self.task_store = (
            task_store
            if isinstance(task_store, FeedTaskStore)
            else FeedTaskStore(task_store)
            if task_store is not None
            else None
        )
        self.catalog = catalog
        self.catalog_path = None if catalog_path in (None, "", "null") else Path(catalog_path)
        self.calibration = calibration
        self.env_factory = env_factory

    async def _handle_processing_tools_state(self, agent_data):
        """Reject parallel calls and map accepted commits to assistant token ends."""

        state = current_feed_runtime_state.get()
        calls = list(getattr(agent_data, "tool_calls", ()) or ())
        if len(calls) > 1:
            if state is not None:
                state["terminate"] = True
                state["infrastructure_invalid"] = True
                state["termination_reason"] = "parallel_tool_calls"
                state["error_type"] = "ParallelToolCalls"
            return AgentState.TERMINATED

        if state is not None and calls and getattr(calls[0], "name", None) == COMMIT_TOOL_NAME:
            # response_mask uses response-relative coordinates and already contains
            # the current assistant generation at this lifecycle point.
            state["pending_commit_position"] = len(agent_data.response_mask) - 1
        try:
            next_state = await super()._handle_processing_tools_state(agent_data)
        finally:
            if state is not None:
                state["pending_commit_position"] = None
        if state is not None and state.get("terminate"):
            return AgentState.TERMINATED
        return next_state

    async def run(self, sampling_params, **kwargs):
        """Resolve a task, execute the native loop, and settle terminal metadata."""

        task = self._resolve_task(kwargs)
        if not self.min_feed_steps <= len(task.episode_seed.videos) <= self.max_feed_steps:
            raise ValueError(
                f"episode has {len(task.episode_seed.videos)} videos; "
                f"required range is {self.min_feed_steps}--{self.max_feed_steps}"
            )
        session = FeedTrajectorySession(
            task,
            max_info_calls_per_video=self.max_info_calls_per_video,
            env_factory=self.env_factory,
        )
        state = await session.start()
        env = session.env
        try:
            self._validate_versions(env.observation())
            output = await super().run(sampling_params, **kwargs)
            if not state["done"]:
                state["terminate"] = True
                state["termination_reason"] = (
                    state["termination_reason"]
                    or "assistant_finished_before_feed_terminal"
                )
            normal_terminal = bool(state["done"] and not state["infrastructure_invalid"])
            episode_return = float(env.total_reward) if normal_terminal else 0.0
            if not math.isfinite(episode_return):
                episode_return = 0.0
                state["infrastructure_invalid"] = True
                state["termination_reason"] = "non_finite_episode_return"

            positions = list(state["assistant_commit_positions"])
            response_length = len(getattr(output, "response_ids", ()) or ())
            counterfactual_values = None
            counterfactual_status = "not_requested"
            if self.credit_mode == "counterfactual" and normal_terminal:
                try:
                    counterfactual_values = _counterfactual_credit(env)
                except Exception as exc:  # keep the factual training path usable
                    counterfactual_status = f"unavailable:{exc.__class__.__name__}"
                else:
                    counterfactual_status = "computed_with_common_random_numbers"

            credit = build_process_credit_metadata(
                env.transitions,
                positions,
                response_length=response_length,
                mode=self.credit_mode,
                gamma=self.credit_gamma,
                episode_return=episode_return,
                counterfactual_values=counterfactual_values,
            )
            credit["counterfactual_status"] = counterfactual_status
            output.reward_score = episode_return
            if getattr(output, "extra_fields", None) is None:
                output.extra_fields = {}
            feed_extra = {
                "episode_id": task.episode_seed.episode_id,
                "versions": {
                    "environment": FEED_ENVIRONMENT_VERSION,
                    "observation": FEED_OBSERVATION_VERSION,
                    "tools": FEED_TOOLS_VERSION,
                    "reward": FEED_REWARD_VERSION,
                },
                "done": bool(state["done"]),
                "termination_reason": state["termination_reason"],
                "infrastructure_invalid": bool(state["infrastructure_invalid"]),
                "tool_calls": len(state["tool_calls"]),
                "commits": len(state["commits"]),
                "guard_rejections": int(state["guard_rejections"]),
                "episode_return": episode_return,
                "credit": credit,
                "metrics": _public_episode_metrics(env),
            }
            output.extra_fields["feed"] = feed_extra
            # The repository's pinned veRL patch consumes a legacy field named
            # ``shopping`` for dynamic group filtering and metrics.  Feed keeps
            # its native diagnostics above and supplies a strict compatibility
            # projection so the patched trainer can actually consume the batch.
            output.extra_fields["shopping"] = _shopping_patch_projection(
                env,
                state,
                episode_return,
                feed_extra["metrics"],
            )
            return output
        finally:
            await session.close()

    def _resolve_task(self, kwargs: Mapping[str, Any]) -> FeedTask:
        extra = _as_mapping(_unwrap_scalar(kwargs.get("extra_info")))
        forbidden_top = {"feed_task", "catalog", "catalog_path"}
        forbidden_extra = forbidden_top | {"catalog_products", "calibration"}
        top_overrides = sorted(
            key for key in forbidden_top if _unwrap_scalar(kwargs.get(key)) is not None
        )
        extra_overrides = sorted(
            key for key in forbidden_extra if _unwrap_scalar(extra.get(key)) is not None
        )
        if top_overrides or extra_overrides:
            raise ValueError(
                "Feed runtime task override fields are forbidden: "
                + ",".join((*top_overrides, *extra_overrides))
            )

        raw_seed = _first_present(kwargs, extra, "episode_seed")
        if raw_seed is not None:
            if kwargs.get("episode_seed") is not None and extra.get("episode_seed") is not None:
                top_seed = _coerce_seed(kwargs["episode_seed"])
                extra_seed = _coerce_seed(extra["episode_seed"])
                if top_seed.to_dict() != extra_seed.to_dict():
                    raise ValueError("duplicate Feed episode_seed values disagree")
            seed = _coerce_seed(raw_seed)
            raw_catalog = _first_present(
                kwargs,
                extra,
                "catalog",
                "catalog_products",
                "catalog_path",
            )
            catalog = (
                _coerce_catalog(raw_catalog)
                if raw_catalog is not None
                else self._default_catalog()
            )
            if catalog is None:
                raise ValueError(
                    "Feed task supplies episode_seed but no catalog/catalog_path"
                )
            calibration = _coerce_calibration(
                _first_present(kwargs, extra, "calibration")
            )
            return FeedTask(seed, catalog, calibration or self.calibration)

        task_key = _first_present(kwargs, extra, "episode_id", "task_id", "index")
        if self.task_store is None or task_key is None:
            raise ValueError(
                "Feed AgentLoop requires episode_seed+catalog or a registered "
                "episode_id/task_id"
            )
        return self.task_store.resolve(_unwrap_scalar(task_key))

    def _default_catalog(self) -> ProductCatalog | None:
        if self.catalog is not None:
            if not isinstance(self.catalog, ProductCatalog):
                self.catalog = _coerce_catalog(self.catalog)
            return self.catalog
        if self.catalog_path is not None:
            self.catalog = _coerce_catalog(self.catalog_path)
            return self.catalog
        return None

    def _validate_versions(self, observation: Mapping[str, Any]) -> None:
        actual_environment = observation.get("environment_version")
        actual_observation = observation.get("observation_version")
        if actual_environment != self.required_environment_version:
            raise RuntimeError(
                "Feed environment version mismatch: "
                f"expected {self.required_environment_version!r}, got {actual_environment!r}"
            )
        if actual_observation != self.required_observation_version:
            raise RuntimeError(
                "Feed observation version mismatch: "
                f"expected {self.required_observation_version!r}, got {actual_observation!r}"
            )
        if self.required_reward_version != FEED_REWARD_VERSION:
            raise RuntimeError("Feed reward version mismatch")
        if self.required_tools_version != FEED_TOOLS_VERSION:
            raise RuntimeError("Feed tools version mismatch")


def build_process_credit_metadata(
    transitions: Sequence[Mapping[str, Any]],
    assistant_commit_positions: Sequence[int | None],
    *,
    response_length: int,
    mode: str,
    gamma: float = 0.99,
    episode_return: float | None = None,
    counterfactual_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build step and token-aligned credit metadata for one trajectory.

    ``response_credit_vector`` is intentionally not written into veRL's reward
    tensor.  Vanilla veRL 0.8 will still use the scalar ``reward_score`` assigned
    by :meth:`FeedToolAgentLoop.run`.
    """

    if mode not in CREDIT_MODES:
        raise ValueError(f"unsupported credit mode: {mode!r}")
    if isinstance(response_length, bool) or int(response_length) < 0:
        raise ValueError("response_length must be non-negative")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and between zero and one")

    step_rewards = [_transition_total(item) for item in transitions]
    total = math.fsum(step_rewards) if episode_return is None else float(episode_return)
    if mode == "terminal":
        step_values = [0.0] * len(step_rewards)
        if step_values:
            step_values[-1] = total
        source = "terminal_episode_return"
    elif mode == "rtg":
        step_values = [0.0] * len(step_rewards)
        running = 0.0
        for index in range(len(step_rewards) - 1, -1, -1):
            running = step_rewards[index] + float(gamma) * running
            step_values[index] = running
        source = "discounted_return_to_go"
    elif mode == "event":
        step_values = _event_source_credit(transitions)
        source = "delayed_event_source_step_attribution"
    else:
        if counterfactual_values is None:
            # Fail closed: never label factual rewards as counterfactual credit.
            step_values = [0.0] * len(step_rewards)
            source = "counterfactual_unavailable"
        else:
            step_values = [float(value) for value in counterfactual_values]
            if len(step_values) != len(step_rewards):
                raise ValueError(
                    "counterfactual_values length must match Feed transitions"
                )
            if not all(math.isfinite(value) for value in step_values):
                raise ValueError("counterfactual_values must be finite")
            source = "crn_replace_one_commit_with_no_recommend"

    positions = [
        int(value) if isinstance(value, int) and not isinstance(value, bool) else None
        for value in assistant_commit_positions
    ]
    token_values = [0.0] * int(response_length)
    mappings = []
    for step, value in enumerate(step_values):
        position = positions[step] if step < len(positions) else None
        mapped = isinstance(position, int) and 0 <= position < len(token_values)
        if mapped:
            token_values[position] += float(value)
        mappings.append(
            {
                "feed_step": step,
                "assistant_commit_position": position,
                "credit": float(value),
                "mapped": bool(mapped),
            }
        )
    return {
        "contract": PROCESS_CREDIT_CONTRACT,
        "mode": mode,
        "gamma": float(gamma),
        "source": source,
        "factual_step_rewards": step_rewards,
        "step_credit": step_values,
        "assistant_commit_positions": positions,
        "commit_position_mapping": mappings,
        "response_credit_vector": token_values,
        "metadata_only": True,
        "native_advantage_integration": False,
        "native_reward_boundary": NATIVE_REWARD_BOUNDARY,
    }


def _event_source_credit(
    transitions: Sequence[Mapping[str, Any]],
) -> list[float]:
    """Move delayed purchase/return value back to the originating commit step."""
    attributed = event_credit_by_source(transitions)
    return [float(attributed.get(step, 0.0)) for step in range(len(transitions))]


def _counterfactual_credit(env: FeedShoppingEnv) -> list[float]:
    """Replay each commit as ``no_recommend`` under the same addressed RNG stream."""

    factual_return = float(env.total_reward)
    transitions = list(env.transitions)
    credits: list[float] = []
    for replaced_step in range(len(transitions)):
        replay = FeedShoppingEnv(
            env.seed,
            env.catalog,
            calibration=env.calibration,
            max_info_calls_per_video=env.max_info_calls_per_video,
            recent_event_limit=env.recent_event_limit,
        )
        for index, transition in enumerate(transitions):
            for tool_record in transition.get("tools", ()):
                name = str(tool_record.get("tool_name", tool_record.get("tool", "")))
                arguments = tool_record.get("arguments", {})
                if name in INFO_TOOL_NAMES:
                    replay.call_tool(name, arguments)
            if index == replaced_step:
                action = {
                    "decision": "no_recommend",
                    "surface": "none",
                    "strategy": "none",
                    "relationship": "primary",
                    "product_ids": [],
                    "evidence_ids": [],
                    "explanation": "CRN counterfactual baseline",
                }
            else:
                action = dict(transition.get("action") or {})
                _repair_replay_evidence(action, replay.observation())
            try:
                replay.step(action)
            except Exception:
                # A prior intervention can change later action legality.  Use the
                # conservative no-intervention continuation rather than peeking at
                # latent state or silently returning a fabricated difference.
                replay.step(
                    {
                        "decision": "no_recommend",
                        "surface": "none",
                        "strategy": "none",
                        "relationship": "primary",
                        "product_ids": [],
                        "evidence_ids": [],
                        "explanation": "counterfactual legality fallback",
                    }
                )
        credits.append(factual_return - float(replay.total_reward))
    return credits


def _repair_replay_evidence(action: MutableMapping[str, Any], observation: Mapping[str, Any]) -> None:
    if str(action.get("decision")) != "recommend":
        return
    available = [str(item) for item in observation.get("evidence_ids", ())]
    retained = [
        str(item)
        for item in action.get("evidence_ids", ())
        if str(item) in set(available)
    ]
    if not any(item.startswith("product.") for item in retained):
        retained.extend(
            [item for item in available if item.startswith("product.")][:1]
        )
    if not any(item.startswith(("video.", "history.", "persona.")) for item in retained):
        retained.extend(
            item
            for item in available
            if item.startswith(("video.", "history.", "persona."))
        )
    action["evidence_ids"] = list(dict.fromkeys(retained))


def _public_episode_metrics(env: FeedShoppingEnv) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    recommendation_count = 0
    no_recommend_count = 0
    for transition in env.transitions:
        decision = str((transition.get("action") or {}).get("decision", ""))
        recommendation_count += int(decision == "recommend")
        no_recommend_count += int(decision == "no_recommend")
        for event in transition.get("events", ()):
            if isinstance(event, Mapping):
                event_type = str(event.get("event_type", "unknown"))
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
    return {
        "steps": len(env.transitions),
        "recommendations": recommendation_count,
        "no_recommendations": no_recommend_count,
        "purchases": event_counts.get("purchase", 0),
        "returns": event_counts.get("return", 0),
        "clicks": event_counts.get("click", 0),
        "cart_additions": event_counts.get("cart", 0),
        "event_counts": dict(sorted(event_counts.items())),
    }


def build_public_tool_payload(
    tool_name: str,
    raw_result: Any,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded public delta instead of replaying the full observation.

    The first user message already contains persona and initial observation data.
    Information calls do not advance the feed, so their response only needs the
    result plus mutable IDs/counters.  A commit additionally carries the next
    video and newly realized public events.  This keeps transcript growth linear
    rather than repeatedly embedding an ever-growing ``recent_events`` list.
    """

    if not isinstance(observation, Mapping):
        raise FeedPublicPayloadError("Feed observation must be an object")
    result = _validate_public_tool_payload(raw_result, "tool_result")
    if not isinstance(result, Mapping):
        raise FeedPublicPayloadError("Feed tool result must be an object")

    state_delta = {
        "step": observation.get("step"),
        "total_steps": observation.get("total_steps"),
        "cart": observation.get("cart", []),
        "purchased_product_ids": observation.get("purchased_product_ids", []),
        "visible_product_ids": observation.get("visible_product_ids", []),
        "evidence_ids": observation.get("evidence_ids", []),
        "info_tool_calls": observation.get("info_tool_calls"),
        "max_info_tool_calls": observation.get("max_info_tool_calls"),
        "done": observation.get("done"),
    }
    payload: dict[str, Any] = {
        "payload_version": FEED_TOOL_PAYLOAD_VERSION,
        "ok": True,
        "tool": str(tool_name),
        "state_delta": state_delta,
    }
    if tool_name == COMMIT_TOOL_NAME:
        # FeedShoppingEnv.call_tool returns a full observation for ordinary
        # in-process clients.  AgentLoop already has that state, so never echo it
        # into the model transcript.
        payload["result"] = {
            "events": result.get("events", []),
            "done": result.get("done", observation.get("done")),
        }
        state_delta.update(
            {
                "observation_version": observation.get("observation_version"),
                "environment_version": observation.get("environment_version"),
                "episode_id": observation.get("episode_id"),
                "current_video": observation.get("current_video"),
            }
        )
    else:
        payload["result"] = result
    return _validate_public_tool_payload(payload)


def _shopping_patch_projection(
    env: FeedShoppingEnv,
    state: Mapping[str, Any],
    episode_return: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Project Feed terminal diagnostics onto the pinned trainer contract.

    The repository's veRL patch predates the Feed profile and hard-requires a
    ``shopping`` record.  Only the scalar terminal utility and explicit success /
    invalid flags affect dynamic group selection; the remaining numeric fields
    are labeled compatibility metrics, not ShopSimulator reward semantics.
    """

    purchases = list(getattr(env, "purchases", ()) or ())
    retained = [purchase for purchase in purchases if not bool(getattr(purchase, "returned", False))]
    qualified = [
        purchase
        for purchase in retained
        if bool(getattr(purchase, "hard_match", False))
        and float(getattr(purchase, "soft_satisfaction", 0.0)) >= 0.5
    ]
    purchase_success = bool(qualified)
    semantic = (
        math.fsum(float(getattr(item, "soft_satisfaction", 0.0)) for item in retained)
        / len(retained)
        if retained
        else 0.0
    )
    hard_match_rate = (
        sum(bool(getattr(item, "hard_match", False)) for item in retained) / len(retained)
        if retained
        else 0.0
    )
    recommendations = int(metrics.get("recommendations", 0) or 0)
    repeat_count = int((metrics.get("event_counts") or {}).get("repeat_exposure", 0) or 0)
    repeat_rate = repeat_count / recommendations if recommendations else 0.0
    evidence_supported = 0
    for transition in getattr(env, "transitions", ()):
        action = transition.get("action") or {}
        if action.get("decision") != "recommend":
            continue
        evidence_ids = [str(item) for item in action.get("evidence_ids", ())]
        products = [str(item) for item in action.get("product_ids", ())]
        product_grounded = all(
            has_product_evidence(evidence_ids, product_id)
            for product_id in products
        )
        context_grounded = any(
            evidence_id.startswith(("video.", "history.", "persona."))
            for evidence_id in evidence_ids
        )
        evidence_supported += int(bool(products) and product_grounded and context_grounded)
    evidence_coverage = (
        evidence_supported / recommendations if recommendations else 0.0
    )
    breakdowns = [
        transition.get("reward") or {}
        for transition in getattr(env, "transitions", ())
    ]
    repeat_penalty = math.fsum(
        float(item.get("repeat_exposure", 0.0)) for item in breakdowns
    )
    unfinished_penalty = 0.0 if bool(state.get("done")) else -1.0
    sampling_invalid = bool(
        state.get("infrastructure_invalid") or not state.get("done")
    )
    steps = len(getattr(env, "transitions", ()) or ())
    terminal_utility = _finite_float(episode_return)
    return {
        "profile": "feed",
        "compatibility_contract": SHOPPING_PATCH_COMPATIBILITY,
        "steps": steps,
        "done": bool(state.get("done")),
        "termination_reason": (
            state.get("termination_reason")
            if state.get("infrastructure_invalid")
            else "feed_exhausted"
            if state.get("done")
            else state.get("termination_reason")
        ),
        "infrastructure_invalid": bool(state.get("infrastructure_invalid")),
        "reward_unverifiable": False,
        "reward_type": "feed_terminal_scalar",
        "reward": {
            "full": float(purchase_success),
            "strict": float(purchase_success),
            "native": terminal_utility,
            "semantic": float(semantic),
            "total": terminal_utility,
            "efficiency": terminal_utility / max(steps, 1),
            "penalty_overlong": 0.0,
            "penalty_unfinished": unfinished_penalty,
            "penalty_repeat": float(repeat_penalty),
            "repeat_action_rate": float(repeat_rate),
            "r_type": 0.0,
            "r_att": float(hard_match_rate),
            "r_option": 0.0,
            "r_price": 0.0,
            "terminal_utility": terminal_utility,
            "purchase_success": purchase_success,
            "sampling_invalid": sampling_invalid,
            "match_score": float(hard_match_rate),
            "evidence_coverage": float(evidence_coverage),
        },
    }


def _coerce_task(value: Any, default_catalog: Any = None) -> FeedTask:
    if isinstance(value, FeedTask):
        return value
    if isinstance(value, Mapping):
        seed = _coerce_seed(value.get("episode_seed", value.get("seed")))
        raw_catalog = value.get(
            "catalog",
            value.get("catalog_products", value.get("catalog_path", default_catalog)),
        )
        catalog = _coerce_catalog(raw_catalog)
        return FeedTask(seed, catalog, _coerce_calibration(value.get("calibration")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) in {2, 3}:
        calibration = _coerce_calibration(value[2]) if len(value) == 3 else None
        return FeedTask(_coerce_seed(value[0]), _coerce_catalog(value[1]), calibration)
    raise TypeError("Feed task must be FeedTask, mapping, or (seed, catalog) tuple")


def _coerce_seed(value: Any) -> EpisodeSeed:
    value = _unwrap_scalar(value)
    if isinstance(value, EpisodeSeed):
        return value
    if isinstance(value, Mapping):
        return EpisodeSeed.from_dict(value)
    if isinstance(value, (str, Path)):
        raw = str(value).strip()
        payload = (
            json.loads(raw)
            if raw.startswith("{")
            else json.loads(Path(value).read_text(encoding="utf-8"))
        )
        return EpisodeSeed.from_dict(payload)
    raise TypeError("episode_seed must be EpisodeSeed, mapping, or JSON path")


def _coerce_catalog(value: Any) -> ProductCatalog:
    value = _unwrap_scalar(value)
    if isinstance(value, ProductCatalog):
        return value
    if isinstance(value, (str, Path)):
        raw = str(value).strip()
        if raw.startswith(("[", "{")):
            return ProductCatalog(json.loads(raw))
        path = Path(value)
        if ".jsonl" in path.suffixes:
            return ProductCatalog.from_jsonl(path)
        return ProductCatalog.from_shopsimulator(path)
    if isinstance(value, Mapping):
        return ProductCatalog(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return ProductCatalog(value)
    raise TypeError("catalog must be ProductCatalog, products, JSONL, or ShopSimulator JSON")


def _coerce_calibration(value: Any) -> Any:
    value = _unwrap_scalar(value)
    if value is None or isinstance(value, Mapping) or hasattr(value, "to_dict"):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{"):
            parsed = json.loads(raw)
            if isinstance(parsed, Mapping):
                return parsed
    raise TypeError("calibration must be a mapping or JSON object string")


def _first_present(*sources_and_names: Any) -> Any:
    names: list[str] = []
    sources: list[Mapping[str, Any]] = []
    for item in sources_and_names:
        if isinstance(item, Mapping) and not names:
            sources.append(item)
        else:
            names.append(str(item))
    for name in names:
        for source in sources:
            if name in source and source[name] is not None:
                return _unwrap_scalar(source[name])
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unwrap_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, Mapping)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


_FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "latent",
    "hidden",
    "probability",
    "propensity",
    "likelihood",
    "reward",
    "hindsight",
    "qualified",
    "hard_match",
    "soft_satisfaction",
    "budget_remaining",
    "purchase_intent",
    "short_term_interest",
    "trust",
    "fatigue",
    "price_sensitivity",
    "oracle",
    "gold_",
)


def _validate_public_tool_payload(value: Any, path: str = "tool_payload") -> Any:
    """Return a detached JSON value, rejecting side-channel-looking keys."""

    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeedPublicPayloadError(f"non-finite value at {path}")
        return value
    if isinstance(value, Mapping):
        result = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise FeedPublicPayloadError(f"non-string key at {path}")
            folded = raw_key.casefold()
            if any(fragment in folded for fragment in _FORBIDDEN_PUBLIC_KEY_FRAGMENTS):
                raise FeedPublicPayloadError(f"sensitive key at {path}.{raw_key}")
            result[raw_key] = _validate_public_tool_payload(item, f"{path}.{raw_key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _validate_public_tool_payload(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise FeedPublicPayloadError(
        f"non-JSON value at {path}: {type(value).__name__}"
    )


def _public_error(code: str, detail: str | None = None) -> ToolResponse:
    payload = {
        "ok": False,
        "error": {
            "code": str(code),
            "detail": None if detail is None else str(detail),
        },
    }
    return ToolResponse(text=_canonical_json(payload))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _transition_total(transition: Mapping[str, Any]) -> float:
    breakdown = transition.get("reward") or transition.get("reward_breakdown") or {}
    if isinstance(breakdown, Mapping) and "total" in breakdown:
        return _finite_float(breakdown["total"])
    if "reward" in transition and not isinstance(transition.get("reward"), Mapping):
        return _finite_float(transition.get("reward"))
    return 0.0


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric credit value, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError("credit values must be finite")
    return result


__all__ = [
    "BaseTool",
    "CREDIT_MODES",
    "FEED_TOOL_PAYLOAD_VERSION",
    "FEED_TOOLS_VERSION",
    "FeedPublicPayloadError",
    "FeedShoppingTool",
    "FeedTask",
    "FeedTaskStore",
    "FeedToolAgentLoop",
    "FeedTrajectorySession",
    "NATIVE_REWARD_BOUNDARY",
    "PROCESS_CREDIT_CONTRACT",
    "SHOPPING_PATCH_COMPATIBILITY",
    "ToolResponse",
    "VERL_AVAILABLE",
    "build_public_tool_payload",
    "build_process_credit_metadata",
    "current_feed_environment",
    "current_feed_runtime_state",
    "make_feed_runtime_state",
]
