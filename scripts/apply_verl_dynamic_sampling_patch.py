#!/usr/bin/env python3
"""Apply or restore the pinned veRL 0.8 shopping-runtime patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
EXPECTED_ORIGINAL_SHA256 = "de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66"
EXPECTED_PATCHED_SHA256 = "fc3564cc5680a9fa92ca7b0a9bc3ae87ccdc90c498ab1bfe34c6796d6c54fb5a"
EXPECTED_ORIGINAL_SCHEMAS_SHA256 = "b5c3a38a44015c27da2dcf1a1f5204f038b895bbf3678c4b1575f62fe1ba5c08"
EXPECTED_PATCHED_SCHEMAS_SHA256 = "8d2dacb132da189dc8a477713a3b934b09288ccf3e79c412fdf2eda0103e4341"
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3"
STRICT_TOOL_SCHEMA_PATCH_MARKER = "SHOPPING_GRPO_STRICT_TOOL_SCHEMA_PATCH_V1"
BACKUP_SUFFIX = ".shopping-grpo-dynamic-sampling.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"
RAY_TRAINER_RELATIVE_PATH = Path("verl/trainer/ppo/ray_trainer.py")
SCHEMAS_RELATIVE_PATH = Path("verl/tools/schemas.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_installed_targets() -> tuple[Path, Path]:
    try:
        installed_version = importlib.metadata.version("verl")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("verl is not installed in the project environment") from exc
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )

    import verl

    verl_source = Path(verl.__file__).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")

    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    schemas = verl_source.parent / "tools" / "schemas.py"
    for target in (ray_trainer, schemas):
        if not target.is_file():
            raise RuntimeError(f"installed veRL patch target does not exist: {target}")
    return ray_trainer.resolve(), schemas.resolve()


def resolve_installed_ray_trainer() -> Path:
    """Compatibility helper retained for callers that inspect the trainer source."""

    return resolve_installed_targets()[0]


def validate_runtime_and_targets(
    target_override: Path | None,
    schemas_target_override: Path | None,
) -> tuple[Path, Path]:
    if target_override is None and schemas_target_override is None:
        return resolve_installed_targets()
    if target_override is None or schemas_target_override is None:
        raise RuntimeError("--target and --schemas-target must be provided together")

    ray_trainer = target_override.resolve()
    schemas = schemas_target_override.resolve()
    for target in (ray_trainer, schemas):
        if not target.is_file():
            raise RuntimeError(f"override patch target does not exist: {target}")
    return ray_trainer, schemas


def _verify_patched_file(
    target: Path,
    *,
    expected_hash: str,
    marker: str,
    label: str,
) -> None:
    target_hash = sha256(target)
    if target_hash != expected_hash:
        raise RuntimeError(
            f"patched {label} hash mismatch: expected {expected_hash}, got {target_hash}"
        )
    if marker not in target.read_text(encoding="utf-8"):
        raise RuntimeError(f"patched {label} is missing marker {marker}")
    py_compile.compile(str(target), doraise=True)


def verify_patched(ray_trainer: Path, schemas: Path) -> None:
    _verify_patched_file(
        ray_trainer,
        expected_hash=EXPECTED_PATCHED_SHA256,
        marker=PATCH_MARKER,
        label="ray_trainer.py",
    )
    _verify_patched_file(
        schemas,
        expected_hash=EXPECTED_PATCHED_SCHEMAS_SHA256,
        marker=STRICT_TOOL_SCHEMA_PATCH_MARKER,
        label="schemas.py",
    )


def _patch_section(relative_path: Path) -> str:
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")
    lines = PATCH_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    header = f"--- a/{relative_path.as_posix()}\n"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise RuntimeError(f"patch file is missing section for {relative_path}") from exc
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("--- a/")),
        len(lines),
    )
    return "".join(lines[start:end])


def _validate_known_state(
    target: Path,
    *,
    original_hash: str,
    patched_hash: str,
    label: str,
) -> str:
    target_hash = sha256(target)
    if target_hash == original_hash:
        return "original"
    if target_hash == patched_hash:
        return "patched"
    raise RuntimeError(
        f"refusing to patch unknown {label}: expected original SHA256 "
        f"{original_hash} or patched SHA256 {patched_hash}, got {target_hash}"
    )


def _ensure_backup(target: Path, expected_original_hash: str) -> Path:
    backup = Path(str(target) + BACKUP_SUFFIX)
    if backup.exists() and sha256(backup) != expected_original_hash:
        raise RuntimeError(f"refusing to overwrite invalid backup: {backup}")
    if not backup.exists():
        shutil.copy2(target, backup)
    return backup


def _apply_file_section(
    patch_program: str,
    target: Path,
    relative_path: Path,
    temporary_directory: Path,
) -> None:
    section_path = temporary_directory / f"{target.name}.patch"
    section_path.write_text(_patch_section(relative_path), encoding="utf-8")
    subprocess.run(
        [patch_program, "--batch", "--forward", "--silent", str(target), str(section_path)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def apply_patch(ray_trainer: Path, schemas: Path) -> None:
    targets = (
        (
            ray_trainer,
            RAY_TRAINER_RELATIVE_PATH,
            EXPECTED_ORIGINAL_SHA256,
            EXPECTED_PATCHED_SHA256,
            "ray_trainer.py",
        ),
        (
            schemas,
            SCHEMAS_RELATIVE_PATH,
            EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            EXPECTED_PATCHED_SCHEMAS_SHA256,
            "schemas.py",
        ),
    )
    states = [
        _validate_known_state(
            target,
            original_hash=original_hash,
            patched_hash=patched_hash,
            label=label,
        )
        for target, _, original_hash, patched_hash, label in targets
    ]
    if states == ["patched", "patched"]:
        verify_patched(ray_trainer, schemas)
        print(f"veRL shopping-runtime patch already applied: {ray_trainer.parent.parent.parent}")
        return

    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("required system 'patch' executable is unavailable")

    for state, (target, _, original_hash, _, _) in zip(states, targets, strict=True):
        if state == "original":
            _ensure_backup(target, original_hash)

    with tempfile.TemporaryDirectory(prefix="verl-shopping-patch-") as temporary:
        temporary_directory = Path(temporary)
        rollback_paths: list[tuple[Path, Path]] = []
        for index, (target, _, _, _, _) in enumerate(targets):
            rollback = temporary_directory / f"rollback-{index}-{target.name}"
            shutil.copy2(target, rollback)
            rollback_paths.append((target, rollback))
        try:
            for state, (target, relative_path, _, _, _) in zip(states, targets, strict=True):
                if state == "original":
                    _apply_file_section(
                        patch_program,
                        target,
                        relative_path,
                        temporary_directory,
                    )
            verify_patched(ray_trainer, schemas)
        except Exception:
            for target, rollback in rollback_paths:
                shutil.copy2(rollback, target)
            raise

    print("applied veRL shopping-runtime patch:")
    for target, _, _, patched_hash, _ in targets:
        print(f"target: {target}")
        print(f"patched_sha256: {patched_hash}")


def restore_patch(ray_trainer: Path, schemas: Path) -> None:
    targets = (
        (
            ray_trainer,
            EXPECTED_ORIGINAL_SHA256,
            EXPECTED_PATCHED_SHA256,
            "ray_trainer.py",
        ),
        (
            schemas,
            EXPECTED_ORIGINAL_SCHEMAS_SHA256,
            EXPECTED_PATCHED_SCHEMAS_SHA256,
            "schemas.py",
        ),
    )
    states = [
        _validate_known_state(
            target,
            original_hash=original_hash,
            patched_hash=patched_hash,
            label=label,
        )
        for target, original_hash, patched_hash, label in targets
    ]
    if states == ["original", "original"]:
        print("veRL shopping-runtime patch targets are already original")
        return

    backups: list[tuple[Path, Path, str]] = []
    for state, (target, original_hash, _, _) in zip(states, targets, strict=True):
        if state == "original":
            continue
        backup = Path(str(target) + BACKUP_SUFFIX)
        if not backup.is_file():
            raise RuntimeError(f"cannot restore without backup: {backup}")
        backup_hash = sha256(backup)
        if backup_hash != original_hash:
            raise RuntimeError(
                f"refusing invalid backup: expected {original_hash}, got {backup_hash}"
            )
        backups.append((target, backup, original_hash))

    with tempfile.TemporaryDirectory(prefix="verl-shopping-restore-") as temporary:
        temporary_directory = Path(temporary)
        rollback_paths: list[tuple[Path, Path]] = []
        for index, (target, _, _, _) in enumerate(targets):
            rollback = temporary_directory / f"rollback-{index}-{target.name}"
            shutil.copy2(target, rollback)
            rollback_paths.append((target, rollback))
        try:
            for target, backup, original_hash in backups:
                restore_temp = target.with_name(target.name + ".shopping-grpo-restore.tmp")
                shutil.copy2(backup, restore_temp)
                restore_temp.replace(target)
                if sha256(target) != original_hash:
                    raise RuntimeError(f"restore verification failed: {target}")
            for target, original_hash, _, _ in targets:
                if sha256(target) != original_hash:
                    raise RuntimeError(f"restore verification failed: {target}")
                py_compile.compile(str(target), doraise=True)
        except Exception:
            for target, rollback in rollback_paths:
                shutil.copy2(rollback, target)
            raise

    print("restored original veRL shopping-runtime files:")
    for target, original_hash, _, _ in targets:
        print(f"target: {target}")
        print(f"original_sha256: {original_hash}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore both verified original files from their automatic backups",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that both targets are already patched without modifying them",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="override ray_trainer.py target for isolated patch-script tests",
    )
    parser.add_argument(
        "--schemas-target",
        type=Path,
        help="override schemas.py target; must be paired with --target",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.restore, args.check)) > 1:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        ray_trainer, schemas = validate_runtime_and_targets(args.target, args.schemas_target)
        if args.restore:
            restore_patch(ray_trainer, schemas)
        elif args.check:
            verify_patched(ray_trainer, schemas)
            print("verified veRL shopping-runtime patch")
        else:
            apply_patch(ray_trainer, schemas)
    except (OSError, RuntimeError, subprocess.CalledProcessError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL shopping-runtime patch error: {exc}") from exc


if __name__ == "__main__":
    main()
