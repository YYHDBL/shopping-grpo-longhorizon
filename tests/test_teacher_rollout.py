import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.teacher_rollout import (
    OpenAIChatClient,
    collect_tasks,
    collect_for_task,
    completed_task_attempts,
    load_tasks,
)


class FakeEnv:
    def __init__(self, **kwargs):
        self.actions = []
        self.released = False

    def reset(self, task_id):
        return {"env_idx": 0, "instruction": f"Instruction: task {task_id}"}

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {"instruction": "results page", "reward": 0.0, "done": False}
        return {
            "instruction": "done page",
            "reward": 1.0,
            "done": True,
            "over": True,
            "purchase": {"asin": "A1"},
            "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
        }

    def release(self):
        self.released = True


class FailingEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        raise RuntimeError("env exploded")


class NonTerminalEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        return {"instruction": "keep going", "reward": 0.0, "done": False}


class MockClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return self.messages.pop(0)


def assistant_tool(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


class TeacherRolloutTest(unittest.TestCase):
    def test_collect_for_task_executes_openai_tool_calls_until_done(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=4,
        )

        self.assertEqual(traj["status"], "done")
        self.assertTrue(traj["trajectory_id"])
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[Buy Now]"])
        self.assertTrue(env.released)
        self.assertEqual(traj["steps"][0]["tool_call"]["function"]["name"], "search_products")
        self.assertEqual(traj["steps"][1]["env_action"], "click[Buy Now]")
        self.assertEqual(traj["terminal_result"]["purchase"]["asin"], "A1")
        self.assertTrue(any(message["role"] == "tool" for message in traj["messages"]))

    def test_collect_for_task_keeps_exception_trajectory_and_releases_env(self):
        client = MockClient([assistant_tool("search_products", {"query": "乳胶枕"})])
        env = FailingEnv()

        traj = collect_for_task(
            {"task_id": 8},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "error")
        self.assertIn("env exploded", traj["error"]["message"])
        self.assertEqual(traj["steps"][0]["env_action"], "search[乳胶枕]")
        self.assertTrue(env.released)

    def test_completed_task_attempts_treats_legacy_rows_as_attempt_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "trajectory_id": "old"}) + "\n"
                + json.dumps({"task_id": 3, "trajectory_id": "old2"}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(completed_task_attempts(output), {(1, 0), (3, 0)})

    def test_collect_tasks_skips_existing_output_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(json.dumps({"task_id": 1, "trajectory_id": "old"}) + "\n")
            client = MockClient([assistant_tool("buy_now", {}, "call_buy")])

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1, 2])
        self.assertEqual(len(written), 1)

    def test_collect_tasks_resumes_missing_attempts_for_each_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "attempt_index": 0, "trajectory_id": "old"}) + "\n",
                encoding="utf-8",
            )
            client = MockClient(
                [
                    assistant_tool("buy_now", {}, "call_1"),
                    assistant_tool("buy_now", {}, "call_2"),
                    assistant_tool("buy_now", {}, "call_3"),
                ]
            )

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                attempts_per_task=2,
            )

        self.assertEqual(
            [(row["task_id"], row["attempt_index"]) for row in written],
            [(1, 1), (2, 0), (2, 1)],
        )

    def test_collect_for_task_caps_multiple_tool_calls_by_max_steps(self):
        client = MockClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "search_products",
                                "arguments": json.dumps({"query": f"乳胶枕{index}"}, ensure_ascii=False),
                            },
                        }
                        for index in range(3)
                    ],
                }
            ]
        )
        env = NonTerminalEnv()

        traj = collect_for_task(
            {"task_id": 9},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "max_steps")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(len(env.actions), 2)

    def test_load_tasks_reads_interaction_task_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = Path(tmpdir) / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "hello"}],
                        "extra_info": {"interaction_kwargs": {"task_id": 42}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_tasks(tasks)

        self.assertEqual(rows[0]["task_id"], 42)
        self.assertEqual(rows[0]["prompt"][0]["content"], "hello")

    def test_openai_client_sends_standard_tool_payload(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-chat",
            base_url="https://api.example.test/v1",
            api_key="secret",
            temperature=0.2,
            top_p=0.9,
            timeout=12,
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(captured["url"], "https://api.example.test/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "deepseek-chat")
        self.assertEqual(captured["payload"]["tools"], [{"type": "function"}])
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")


if __name__ == "__main__":
    unittest.main()
