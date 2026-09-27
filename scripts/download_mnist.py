import argparse
import sys

from minidl.data.mnist import download_mnist, load_mnist


def main():
    parser = argparse.ArgumentParser(
        description="Download verified MNIST IDX files and build NumPy cache"
    )
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    try:
        download_mnist(args.data_root)
        for split in ("train", "test"):
            dataset = load_mnist(args.data_root, split=split)
            print(split, dataset.x.shape, dataset.y.shape)
    except (OSError, ValueError) as error:
        print(f"MNIST download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
