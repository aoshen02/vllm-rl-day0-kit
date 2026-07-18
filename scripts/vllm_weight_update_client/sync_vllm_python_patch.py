"""Copy an exact-base vLLM Python patch into a stopped Docker container.

The container image and host worktree must resolve to the same vLLM commit.
Only dirty ``vllm/**/*.py`` files are accepted. Before any overwrite, every
container file is compared with the worktree's ``HEAD`` version; after copying,
it is compared with the dirty host file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


IMAGE_PROBE = """
import json
from pathlib import Path

import vllm
from vllm import _version

package_root = Path(vllm.__file__).resolve().parent
print(json.dumps({
    "version": vllm.__version__,
    "commit_id": _version.__commit_id__,
    "package_root": str(package_root),
}, sort_keys=True))
"""


def _run(
    command: list[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=text)
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode().strip()
        raise RuntimeError(f"Command failed: {command!r}\n{stderr}")
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_output(worktree: Path, *args: str) -> str:
    return _run(["git", "-C", str(worktree), *args]).stdout.strip()


def _image_inspect(image: str) -> dict[str, Any]:
    return json.loads(_run(["docker", "image", "inspect", image]).stdout)[0]


def _container_inspect(container: str) -> dict[str, Any]:
    return json.loads(_run(["docker", "container", "inspect", container]).stdout)[0]


def _probe_image(image: str) -> dict[str, str]:
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/usr/bin/python3",
            image,
            "-c",
            IMAGE_PROBE,
        ]
    )
    return json.loads(result.stdout)


def _normalize_file(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"File must be a worktree-relative path: {value}")
    if not path.parts or path.parts[0] != "vllm" or path.suffix != ".py":
        raise ValueError(f"Only vllm/**/*.py files can be synchronized: {value}")
    return path


def _dirty_python_files(worktree: Path) -> set[PurePosixPath]:
    changed = _git_output(
        worktree,
        "diff",
        "--name-only",
        "--diff-filter=ACMRTUXB",
        "HEAD",
        "--",
        "vllm",
    ).splitlines()
    untracked = _git_output(
        worktree,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "vllm",
    ).splitlines()
    deleted = _git_output(
        worktree,
        "diff",
        "--name-only",
        "--diff-filter=D",
        "HEAD",
        "--",
        "vllm",
    ).splitlines()
    deleted_python = [path for path in deleted if path.endswith(".py")]
    if deleted_python:
        raise RuntimeError("Deleting container files is intentionally unsupported: " + ", ".join(deleted_python))
    return {PurePosixPath(path) for path in changed + untracked if path.endswith(".py")}


def _head_file(worktree: Path, path: PurePosixPath) -> bytes | None:
    object_name = f"HEAD:{path.as_posix()}"
    exists = _run(
        ["git", "-C", str(worktree), "cat-file", "-e", object_name],
        check=False,
    )
    if exists.returncode != 0:
        return None
    return _run(
        ["git", "-C", str(worktree), "show", object_name],
        text=False,
    ).stdout


def _copy_from_container(
    container: str,
    source: PurePosixPath,
    destination: Path,
) -> bytes | None:
    result = _run(
        ["docker", "cp", f"{container}:{source.as_posix()}", str(destination)],
        check=False,
    )
    if result.returncode == 0:
        return destination.read_bytes()
    message = result.stderr.lower()
    if "could not find" in message or "no such file" in message:
        return None
    raise RuntimeError(f"Cannot read {container}:{source}: {result.stderr.strip()}")


def _write_manifest(path: Path | None, manifest: dict[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    print(payload, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--container", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--file", action="append")
    selection.add_argument(
        "--all-dirty",
        action="store_true",
        help="Synchronize the exact set of dirty vllm/**/*.py files",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    if Path(_git_output(worktree, "rev-parse", "--show-toplevel")) != worktree:
        raise RuntimeError(f"--worktree must be the git top level: {worktree}")

    dirty_files = _dirty_python_files(worktree)
    files = (
        dirty_files
        if args.all_dirty
        else {_normalize_file(value) for value in args.file or []}
    )
    if files != dirty_files:
        missing = sorted(str(path) for path in dirty_files - files)
        extra = sorted(str(path) for path in files - dirty_files)
        raise RuntimeError(f"--file must exactly cover dirty vLLM Python files; missing={missing}, not_dirty={extra}")

    image_info = _image_inspect(args.image)
    container_info = _container_inspect(args.container)
    image_id = image_info["Id"]
    if container_info["Image"] != image_id:
        raise RuntimeError(f"Container image {container_info['Image']} != requested image {image_id}")
    status = container_info["State"]["Status"]
    if status not in {"created", "exited"}:
        raise RuntimeError(f"Refusing to patch container in state {status!r}; use created/exited")

    image_probe = _probe_image(args.image)
    head = _git_output(worktree, "rev-parse", "HEAD")
    image_commit = image_probe["commit_id"].removeprefix("g")
    if not head.startswith(image_commit):
        raise RuntimeError(f"Worktree HEAD {head} does not match image commit {image_commit}")

    package_root = PurePosixPath(image_probe["package_root"])
    if not package_root.is_absolute() or package_root.name != "vllm":
        raise RuntimeError(f"Unexpected image package root: {package_root}")

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vllm-container-sync-") as temp:
        temp_root = Path(temp)
        for index, relative in enumerate(sorted(files)):
            host_path = worktree / relative.as_posix()
            if not host_path.is_file() or host_path.is_symlink():
                raise RuntimeError(f"Patch source must be a regular file: {host_path}")

            baseline = _head_file(worktree, relative)
            container_path = package_root / PurePosixPath(*relative.parts[1:])
            container_before = _copy_from_container(
                args.container,
                container_path,
                temp_root / f"before-{index}.py",
            )
            if baseline is None and container_before is not None:
                raise RuntimeError(f"New host file unexpectedly exists in container: {relative}")
            if baseline is not None and container_before is None:
                raise RuntimeError(f"Baseline file is absent in container: {relative}")
            if baseline is not None and container_before != baseline:
                raise RuntimeError(f"Container baseline differs from worktree HEAD: {relative}")

            current = host_path.read_bytes()
            records.append(
                {
                    "path": relative.as_posix(),
                    "container_path": container_path.as_posix(),
                    "baseline_sha256": _sha256(baseline) if baseline else None,
                    "patch_sha256": _sha256(current),
                    "container_before_sha256": (_sha256(container_before) if container_before else None),
                }
            )

        if not args.dry_run:
            for record in records:
                source = worktree / record["path"]
                destination = f"{args.container}:{record['container_path']}"
                _run(["docker", "cp", str(source), destination])

            for index, record in enumerate(records):
                after = _copy_from_container(
                    args.container,
                    PurePosixPath(record["container_path"]),
                    temp_root / f"after-{index}.py",
                )
                after_hash = _sha256(after) if after is not None else None
                record["container_after_sha256"] = after_hash
                if after_hash != record["patch_sha256"]:
                    raise RuntimeError(f"Post-copy hash mismatch for {record['path']}: {after_hash}")

    manifest = {
        "schema": "vllm-agent-infra.container_python_sync.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run" if args.dry_run else "copy",
        "image": args.image,
        "image_id": image_id,
        "image_repo_digests": image_info.get("RepoDigests", []),
        "image_vllm_version": image_probe["version"],
        "image_vllm_commit": image_probe["commit_id"],
        "worktree": str(worktree),
        "worktree_head": head,
        "container": args.container,
        "container_initial_state": status,
        "files": records,
    }
    _write_manifest(args.manifest, manifest)


if __name__ == "__main__":
    main()
