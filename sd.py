#!/usr/bin/env python3
"""
sd.py — superdoc dev runner (Linux / macOS / Windows)

Usage:
  python sd.py build
  python sd.py test                                           # all tests
  python sd.py test src/tests/test_merge.py                  # one file
  python sd.py test src/tests/test_merge.py::TestFullMerge   # one class
  python sd.py test src/tests/test_merge.py::TestFullMerge::test_full_merge_basic_text
  python sd.py test -k tree                                  # keyword filter
  python sd.py shell                                         # bash in container
"""

import subprocess
import sys
import os
from pathlib import Path

IMAGE = "superdoc-dev"
LAMBDA_TASK_ROOT = "/var/task"


def run(cmd: list[str], check=True):
    """Run a command, streaming output live."""
    print(f"» {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode


def build():
    print(f"Building {IMAGE}...")
    run(["docker", "build", "-f", "Dockerfile.dev", "-t", IMAGE, "."])


def image_exists() -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True
    )
    return result.returncode == 0


def ensure_image():
    if not image_exists():
        print(f"Image '{IMAGE}' not found, building first...")
        build()


def docker_run(cmd: list[str]):
    """
    Run a command inside the dev container with all necessary mounts.
    Uses cross-platform path handling.
    """
    project_root = Path.cwd()

    env_file         = project_root / ".env"
    aws_dir          = Path.home() / ".aws"
    test_docs        = project_root / "src" / "tests" / "test_docs.json"

    # On Windows, Docker Desktop accepts both / and \ but prefers /
    def to_docker_path(p: Path) -> str:
        return str(p).replace("\\", "/")

    mounts = []

    if env_file.exists():
        mounts += ["-v", f"{to_docker_path(env_file)}:{LAMBDA_TASK_ROOT}/.env"]
    else:
        print(f"[WARN] .env not found at {env_file} — container may lack credentials")

    if aws_dir.exists():
        mounts += ["-v", f"{to_docker_path(aws_dir)}:/root/.aws:ro"]
    else:
        print(f"[WARN] ~/.aws not found — AWS calls inside container will fail")

    # Mount test_docs.json so created doc IDs persist back to the host.
    # Create it if it doesn't exist so the mount doesn't fail.
    if not test_docs.exists():
        test_docs.parent.mkdir(parents=True, exist_ok=True)
        test_docs.write_text('{"files": {}}')
        print(f"Created empty test_docs.json at {test_docs}")
    mounts += ["-v", f"{to_docker_path(test_docs)}:{LAMBDA_TASK_ROOT}/src/tests/test_docs.json"]

    docker_cmd = ["docker", "run", "--rm", "-it"] + mounts + [IMAGE] + cmd
    run(docker_cmd, check=False)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        build()

    elif command == "test":
        ensure_image()
        # Everything after "test" is forwarded directly to pytest
        pytest_args = sys.argv[2:] if len(sys.argv) > 2 else ["src/tests"]
        docker_run(["pytest"] + pytest_args + ["-s"])

    elif command == "shell":
        ensure_image()
        docker_run(["bash"])

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()