import tempfile
import unittest
from pathlib import Path

from scripts.run_sft_curriculum import build_stage_commands


class SftCurriculumTest(unittest.TestCase):
    def test_builds_three_sequential_train_and_merge_stages(self):
        manifest = {
            "stages": {
                "a": {"epochs": 1.0, "learning_rate": 1e-4},
                "b": {"epochs": 1.0, "learning_rate": 7e-5},
                "c": {"epochs": 1.0, "learning_rate": 5e-5},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            commands = build_stage_commands(
                manifest,
                manifest_path=root / "manifest.json",
                source=root / "all.jsonl",
                base_model="Qwen/Qwen3.5-2B",
                output_root=root / "outputs",
                python="python",
            )

        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0]["stage"], "a")
        self.assertEqual(commands[0]["train"][commands[0]["train"].index("--model") + 1], "Qwen/Qwen3.5-2B")
        self.assertTrue(commands[1]["train"][commands[1]["train"].index("--model") + 1].endswith("stage-a/merged"))
        self.assertTrue(commands[2]["merge"][commands[2]["merge"].index("--output") + 1].endswith("stage-c/merged"))
        self.assertIn("--curriculum-manifest", commands[0]["train"])
        self.assertIn("--gradient-checkpointing", commands[0]["train"])

    def test_start_and_stop_select_a_contiguous_stage_range(self):
        manifest = {
            "stages": {
                stage: {"epochs": 1.0, "learning_rate": 1e-4}
                for stage in ("a", "b", "c")
            }
        }
        commands = build_stage_commands(
            manifest,
            manifest_path=Path("manifest.json"),
            source=Path("all.jsonl"),
            base_model="base",
            output_root=Path("outputs"),
            python="python",
            start_stage="b",
            stop_after_stage="b",
            swanlab=True,
        )

        self.assertEqual([command["stage"] for command in commands], ["b"])
        self.assertIn("--swanlab", commands[0]["train"])
        self.assertIn("stage-b/adapter", " ".join(commands[0]["train"]))

        with self.assertRaisesRegex(ValueError, "after"):
            build_stage_commands(
                manifest,
                manifest_path=Path("manifest.json"),
                source=Path("all.jsonl"),
                base_model="base",
                output_root=Path("outputs"),
                python="python",
                start_stage="c",
                stop_after_stage="a",
            )


if __name__ == "__main__":
    unittest.main()
