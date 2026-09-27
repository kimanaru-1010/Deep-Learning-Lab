"""Create original toy prose offline, or download a user-selected HTTPS corpus."""

import argparse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Optional HTTPS URL; respect the corpus license")
    parser.add_argument("--output", default="data/raw/toy.txt")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        print(f"Using existing {destination}")
        return
    if args.url:
        if not args.url.startswith("https://"):
            parser.error("Only HTTPS URLs are supported")
        with urllib.request.urlopen(args.url, timeout=30) as response:
            text = response.read().decode("utf-8")
    else:
        text = (
            "A small robot learns to read. Each day it studies a line of text. "
            "The sun rises, the rain falls, and the robot writes a new story.\n"
        ) * 100
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(f"Saved {len(text)} characters to {destination}")


if __name__ == "__main__":
    main()
