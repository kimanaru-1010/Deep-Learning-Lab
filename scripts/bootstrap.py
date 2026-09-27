"""Cross-platform bootstrap; never install interpreters or overwrite an existing venv."""

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-install", action="store_true", help="Verify already installed dependencies offline"
    )
    parser.add_argument(
        "--upgrade-pip", action="store_true", help="Explicitly opt in to upgrading pip inside .venv"
    )
    args = parser.parse_args()
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        parser.error("Use existing Python 3.12 or 3.11. No system Python will be installed.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not environment.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    if not python.exists() or not (environment / "pyvenv.cfg").exists():
        parser.error(
            "Existing .venv is incomplete or belongs to another OS. Move it aside manually; nothing was deleted."
        )
    # Validate before any pip writes; use the environment being installed into.
    check = "import runpy; runpy.run_path('scripts/verify_env.py')['check_interpreter']()"
    subprocess.run([str(python), "-c", check], cwd=ROOT, check=True)
    if args.upgrade_pip:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT, check=True
        )
    if not args.skip_install:
        subprocess.run([str(python), "-m", "pip", "install", "-e", ".[dev]"], cwd=ROOT, check=True)
    for directory in (
        "data/raw",
        "data/processed",
        "outputs/plots",
        "outputs/logs",
        "outputs/checkpoints",
        "outputs/runs",
        "outputs/environment",
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), "scripts/verify_env.py", "--record"], cwd=ROOT, check=True)
    subprocess.run([str(python), "-m", "ruff", "check", "."], cwd=ROOT, check=True)
    print(f"PASS setup. Interpreter: {python}")
    print(
        "Next: activate .venv, read docs/curriculum.md, then python experiments/00_manual_backprop/run.py --help"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL setup: {error}", file=sys.stderr)
        raise SystemExit(1) from None
