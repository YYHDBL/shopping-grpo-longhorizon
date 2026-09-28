"""End-to-end test for the public CLI from a non-editable wheel install."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("build"),
    "the wheel test requires the dev extra",
)
class WheelInstallTest(unittest.TestCase):
    def test_wheel_public_cli_runs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dist = temporary / "dist"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    str(dist),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            wheel = next(dist.glob("shopping_grpo-*.whl"))
            environment = temporary / "venv"
            binary_directory = "Scripts" if os.name == "nt" else "bin"
            python = environment / binary_directory / "python"
            cli = environment / binary_directory / "shopping-grpo"
            uv = shutil.which("uv")
            if uv:
                subprocess.run(
                    [uv, "venv", str(environment)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-deps",
                        str(wheel),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                venv.EnvBuilder(with_pip=True).create(environment)
                pip = environment / binary_directory / "pip"
                subprocess.run(
                    [str(pip), "install", "--no-deps", str(wheel)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            trajectory = temporary / "non_blind_example.jsonl"
            trajectory.write_bytes(
                (root / "examples/trajectories.jsonl").read_bytes()
            )
            evaluated = subprocess.run(
                [str(cli), "evaluate", str(trajectory)],
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(evaluated.stdout)["trajectory_count"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
