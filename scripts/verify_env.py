"""Verify the lab environment without installing packages or modifying Python."""

import argparse
import ast
import importlib
import importlib.metadata
import json
import os
import re
import site
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
FORBIDDEN = {
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "jax",
    "jaxlib",
    "flax",
    "mxnet",
    "paddle",
    "paddlepaddle",
    "tinygrad",
    "autograd",
    "micrograd",
    "scikit-learn",
    "sklearn",
    "scipy",
}


def forbidden_name(name):
    normalized = name.lower().replace("_", "-")
    return any(normalized == item or normalized.startswith(item + "-") for item in FORBIDDEN)


def check_interpreter():
    expected = ROOT / ".venv"
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError("Use this repository's .venv Python, not global Python")
    # Do not resolve executable symlinks on Linux: venv/bin/python may target base Python.
    if not Path(sys.executable).absolute().is_relative_to(expected.absolute()):
        raise RuntimeError("Executable is outside repository .venv")
    config = (expected / "pyvenv.cfg").read_text(encoding="utf-8")
    if not re.search(r"include-system-site-packages\s*=\s*false", config, re.I):
        raise RuntimeError("venv must not inherit system site-packages")
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or site.ENABLE_USER_SITE:
        raise RuntimeError("Need isolated Python 3.11/3.12")
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def audit():
    problems = []
    distributions = sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    )
    for name, _ in distributions:
        if forbidden_name(name):
            problems.append(f"Forbidden installed distribution: {name}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        dependencies.extend(extra)
    for dependency in dependencies:
        name = re.split(r"[<>=!~\[; ]", dependency)[0]
        if forbidden_name(name):
            problems.append(f"Forbidden dependency: {name}")
    for folder in ("src", "models", "experiments", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if forbidden_name(name.split(".")[0]):
                        problems.append(
                            f"Forbidden import: {path.relative_to(ROOT)}:{node.lineno}: {name}"
                        )
    if problems:
        raise RuntimeError("\n".join(problems))
    return distributions


def verify(skip_tests=False):
    report = check_interpreter()
    print(f"PASS isolated interpreter: {report['executable']} ({report['python']})")
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, check=True)
    report["pip_check"] = "PASS"
    for name in ("numpy", "matplotlib", "pytest", "tqdm", "minidl"):
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        expected = ROOT / "src" if name == "minidl" else ROOT / ".venv"
        if not source.is_relative_to(expected.resolve()):
            raise RuntimeError(f"{name} imported from unexpected location: {source}")
        report[name] = getattr(module, "__version__", "unknown")
        print(f"PASS {name} {report[name]}: {source}")
    packages = audit()
    report["forbidden_packages_and_imports"] = "NONE"
    print("PASS forbidden dependency / installed package / runtime import audit")
    import numpy as np

    from minidl.utils.gradcheck import check_gradient
    from minidl.utils.visualization import plot_confusion_matrix

    gradient = check_gradient(lambda x: np.sum(x**2), np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    if not gradient.passed:
        raise RuntimeError(str(gradient))
    print(gradient)
    with tempfile.TemporaryDirectory(prefix="minidl-plot-") as directory:
        path = plot_confusion_matrix(np.eye(2), Path(directory) / "smoke.png")
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Invalid Matplotlib PNG")
    report["matplotlib_png"] = "PASS"
    print("PASS Matplotlib headless PNG")
    if not skip_tests:
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-m", "not student", "-q"],
            cwd=ROOT,
            check=True,
        )
        report["infrastructure_tests"] = "PASS"
    else:
        report["infrastructure_tests"] = "NOT RUN (--skip-tests)"
    return report, packages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests", action="store_true", help="Fast audit; full verification is the default"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write version snapshot and report into outputs/environment",
    )
    args = parser.parse_args()
    try:
        report, packages = verify(args.skip_tests)
        if args.record:
            directory = ROOT / "outputs" / "environment"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "verification.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            (directory / "installed-packages.txt").write_text(
                "\n".join(f"{name}=={version}" for name, version in packages) + "\n",
                encoding="utf-8",
            )
        print("PASS environment verification" + (" (tests omitted)" if args.skip_tests else ""))
    except (RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as error:
        print(f"FAIL environment verification: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
