#!/usr/bin/env python3
"""
sd.py — superdoc dev runner (Linux / macOS / Windows)

Usage:
  python sd.py build
  python sd.py test                                           # clear docs & run all tests
  python sd.py test src/tests/test_merge.py                  # clear docs & run one file
  python sd.py test --no-clear src/tests/test_merge.py       # bypass clear & run test file
  python sd.py clear                                         # standalone clear of tracked google docs
  python sd.py shell                                         # bash in container
  python sd.py docs                                          # list tracked google doc links
"""

import json
import os
import subprocess
import sys
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
        ["docker", "image", "inspect", IMAGE], capture_output=True
    )
    return result.returncode == 0


def ensure_image():
    build()


def docker_run(cmd: list[str]):
    """Run a command inside the dev container with all necessary mounts."""
    project_root = Path.cwd()

    env_file = project_root / ".env"
    aws_dir = Path.home() / ".aws"
    test_docs = project_root / "src" / "tests" / "test_docs.json"

    def to_docker_path(p: Path) -> str:
        return str(p).replace("\\", "/")

    mounts = ["-v", f"{to_docker_path(project_root)}:{LAMBDA_TASK_ROOT}"]

    if not env_file.exists():
        print(
            f"[WARN] .env not found at {env_file} — container may lack credentials"
        )

    if aws_dir.exists():
        mounts += ["-v", f"{to_docker_path(aws_dir)}:/root/.aws:ro"]
    else:
        print(f"[WARN] ~/.aws not found — AWS calls inside container will fail")

    if not test_docs.exists():
        test_docs.parent.mkdir(parents=True, exist_ok=True)
        test_docs.write_text('{"files": {}}')
        print(f"Created empty test_docs.json at {test_docs}")

    docker_cmd = ["docker", "run", "--rm", "-it"] + mounts + [IMAGE] + cmd
    run(docker_cmd, check=False)


def clear_docs():
    """Triggers the Python clear script contextually inside the container."""
    ensure_image()
    docker_run(["python", "-m", "src.tests.clear_docs"])


def show_docs():
    """Read, parse, and build clickable Google Doc links from test_docs.json."""
    test_docs = Path.cwd() / "src" / "tests" / "test_docs.json"

    if not test_docs.exists():
        print(
            "[INFO] No test_docs.json file found. Run your tests first to generate document links."
        )
        return

    try:
        with open(test_docs, "r", encoding="utf-8") as f:
            data = json.load(f)

        files = data.get("files", {})
        if not files:
            print(
                "[INFO] test_docs.json is empty. Run tests to populate Google Doc tracking data."
            )
            return

        print("\n📂 Tracked Google Doc Links:")
        print("=" * 80)
        for key, val in files.items():
            if isinstance(val, dict):
                raw_val = (
                    val.get("id") or val.get("url") or val.get("link") or ""
                )
            else:
                raw_val = str(val)

            raw_val = raw_val.strip()

            if not raw_val:
                url = "Unknown/Empty ID"
            elif raw_val.startswith("http://") or raw_val.startswith(
                "https://"
            ):
                url = raw_val
            else:
                url = f"https://docs.google.com/document/d/{raw_val}/edit"

            print(f"📄 {key}:")
            print(f"   {url}\n")
        print("=" * 80 + "\n")

    except json.JSONDecodeError:
        print("[ERROR] test_docs.json is malformed or contains invalid JSON.")
    except Exception as e:
        print(f"[ERROR] Could not read document links: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        build()

    elif command == "test":
        ensure_image()

        # Isolate pytest arguments forwarded from the CLI
        pytest_args = sys.argv[2:] if len(sys.argv) > 2 else ["src/tests"]

        # Catch bypass flag to avoid clearing when you don't want to
        if "--no-clear" in pytest_args:
            pytest_args.remove("--no-clear")
            print("⚡ Skipping Google Doc clearing phase...")
        else:
            clear_docs()

        docker_run(["python", "-m", "pytest"] + pytest_args + ["-s"])

    elif command == "clear":
        clear_docs()

    elif command == "shell":
        ensure_image()
        docker_run(["bash"])

    elif command == "docs":
        show_docs()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()