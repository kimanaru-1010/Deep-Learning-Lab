"""List generated files by default; delete files in one named run only with --apply."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = (Path(__file__).resolve().parents[1] / "outputs" / "runs").resolve()
    target = (root / args.run_name).resolve()
    if target.parent != root or target == root:
        parser.error("Choose one immediate run directory inside outputs/runs")
    if not target.is_dir():
        parser.error("Run does not exist")
    files = list(target.rglob("*"))
    if any(p.is_symlink() or not p.resolve().is_relative_to(target) for p in files):
        parser.error("Refusing to follow links outside the run")
    for path in files:
        if path.is_file():
            print(path.relative_to(root))
            if args.apply:
                path.unlink()
    print("Applied" if args.apply else "Dry run. Add --apply to delete the listed files.")


if __name__ == "__main__":
    main()
