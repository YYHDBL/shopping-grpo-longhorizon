#!/usr/bin/env python3
"""Apply or restore the pinned veRL 0.8 dynamic-sampling patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66"
EXPECTED_PATCHED_SHA256 = "684b491e20ba9d41e91d5010186d4d08b01a01fc67f8a77d17c086b0381e00a3"
SUPERSEDED_PATCHED_SHA256 = "fc3564cc5680a9fa92ca7b0a9bc3ae87ccdc90c498ab1bfe34c6796d6c54fb5a"
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3"
BACKUP_SUFFIX = ".shopping-grpo-dynamic-sampling.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_installed_ray_trainer() -> Path:
    installed_version = importlib.metadata.version("verl")
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )

    import verl

    verl_source = Path(verl.__file__).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")

    target = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not target.is_file():
        raise RuntimeError(f"installed ray_trainer.py does not exist: {target}")
    return target.resolve()


def validate_runtime_and_target(target_override: Path | None) -> Path:
    installed_target = resolve_installed_ray_trainer()
    if target_override is None:
        return installed_target
    target = target_override.resolve()
    if not target.is_file():
        raise RuntimeError(f"target ray_trainer.py does not exist: {target}")
    return target


def verify_patched(target: Path) -> None:
    target_hash = sha256(target)
    if target_hash != EXPECTED_PATCHED_SHA256:
        raise RuntimeError(
            "patched ray_trainer.py hash mismatch: "
            f"expected {EXPECTED_PATCHED_SHA256}, got {target_hash}"
        )
    if PATCH_MARKER not in target.read_text(encoding="utf-8"):
        raise RuntimeError(f"patched ray_trainer.py is missing marker {PATCH_MARKER}")
    py_compile.compile(str(target), doraise=True)


def apply_patch(target: Path) -> None:
    target_hash = sha256(target)
    if target_hash == EXPECTED_PATCHED_SHA256:
        verify_patched(target)
        print(f"veRL dynamic-sampling patch already applied: {target}")
        return
    backup = Path(str(target) + BACKUP_SUFFIX)
    if target_hash == SUPERSEDED_PATCHED_SHA256:
        if not backup.is_file() or sha256(backup) != EXPECTED_ORIGINAL_SHA256:
            raise RuntimeError(
                "cannot upgrade the previous patch without its verified original backup"
            )
        shutil.copy2(backup, target)
        target_hash = EXPECTED_ORIGINAL_SHA256
    if target_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            "refusing to patch unknown ray_trainer.py: "
            f"expected original SHA256 {EXPECTED_ORIGINAL_SHA256}, got {target_hash}"
        )
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")

    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("required system 'patch' executable is unavailable")

    if backup.exists() and sha256(backup) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to overwrite invalid backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)

    rollback_source = backup

    try:
        subprocess.run(
            [patch_program, "--batch", "--forward", "--silent", str(target), str(PATCH_FILE)],
            check=True,
            cwd=PROJECT_ROOT,
        )
        verify_patched(target)
    except Exception:
        shutil.copy2(rollback_source, target)
        raise

    print(f"applied veRL dynamic-sampling patch: {target}")
    print(f"backup: {backup}")
    print(f"patched_sha256: {sha256(target)}")


def restore_patch(target: Path) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    target_hash = sha256(target)
    if target_hash == EXPECTED_ORIGINAL_SHA256:
        print(f"veRL ray_trainer.py is already original: {target}")
        return
    if not backup.is_file():
        raise RuntimeError(f"cannot restore without backup: {backup}")
    backup_hash = sha256(backup)
    if backup_hash != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing invalid backup: expected {EXPECTED_ORIGINAL_SHA256}, got {backup_hash}"
        )

    restore_temp = target.with_name(target.name + ".shopping-grpo-restore.tmp")
    shutil.copy2(backup, restore_temp)
    restore_temp.replace(target)
    if sha256(target) != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"restore verification failed: {target}")
    py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL ray_trainer.py: {target}")
    print(f"original_sha256: {sha256(target)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore the verified original file from the automatic backup",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the target is already patched without modifying it",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="override ray_trainer.py target for isolated patch-script tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.restore, args.check)) > 1:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        target = validate_runtime_and_target(args.target)
        if args.restore:
            restore_patch(target)
        elif args.check:
            verify_patched(target)
            print(f"verified veRL dynamic-sampling patch: {target}")
        else:
            apply_patch(target)
    except (OSError, RuntimeError, subprocess.CalledProcessError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL dynamic-sampling patch error: {exc}") from exc


if __name__ == "__main__":
    main()
