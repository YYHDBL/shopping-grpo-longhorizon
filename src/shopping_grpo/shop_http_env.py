"""Structured client for one leased ShopSimulator environment."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ShopHttpError(RuntimeError):
    """The HTTP request did not reach a usable ShopSimulator response."""


class ShopEnvironmentError(RuntimeError):
    """ShopSimulator accepted HTTP request but reported an environment error."""


class ShopProtocolError(RuntimeError):
    """ShopSimulator response did not match its structured API contract."""


class ShopEnvironmentStateError(RuntimeError):
    """The client lifecycle was used out of order."""


class ShopAgentEnv:
    """One trajectory's exclusive ShopSimulator API lease."""

    def __init__(self, base_url="http://127.0.0.1:5000", timeout=60, transport=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.env_idx = None
        self.done = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.release()
        except Exception:
            if exc_type is None:
                raise
        return False

    def reset(self, task_id):
        if self.env_idx is not None:
            raise ShopEnvironmentStateError("Environment is already leased; release it before reset")

        result = self._call({"action": "reset", "idx": int(task_id)})
        env_idx = result.get("env_idx")
        if not isinstance(env_idx, int):
            raise ShopProtocolError("reset response is missing integer env_idx")
        self.env_idx = env_idx
        self.done = False
        return result

    def step(self, action):
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        if self.done:
            raise ShopEnvironmentStateError("Environment is already done; release it before reset")
        result = self._call(
            {"action": "interact", "env_idx": self._leased_env_idx(), "response": action}
        )
        self.done = bool(result.get("done", False))
        return result

    def release(self):
        if self.env_idx is None:
            return None

        env_idx = self.env_idx
        try:
            return self._call({"action": "release_one", "env_idx": env_idx})
        finally:
            self.env_idx = None
            self.done = False

    def _leased_env_idx(self):
        if self.env_idx is None:
            raise ShopEnvironmentStateError("reset must succeed before step")
        return self.env_idx

    def _call(self, payload):
        try:
            response = self._send(payload)
        except (HTTPError, URLError, OSError) as exc:
            raise ShopHttpError(f"ShopSimulator HTTP request failed: {exc}") from exc

        if not isinstance(response, dict):
            raise ShopProtocolError("ShopSimulator response must be a JSON object")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ShopProtocolError("ShopSimulator response is missing object result")
        if result.get("error"):
            raise ShopEnvironmentError(str(result["error"]))
        return result

    def _send(self, payload):
        endpoint = f"{self.base_url}/api/shop_agent"
        if self.transport is not None:
            return self.transport(endpoint, payload, self.timeout)

        body = json.dumps(payload).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "shopping-grpo/0.2"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
