"""Command-line entry point for training and conditioned sampling."""

import argparse
from importlib.metadata import version


def main(argv: list[str] | None = None) -> None:
    """Dispatch a command, preserving argparse errors and process exit status."""
    parser = argparse.ArgumentParser(description="Annotation Assisted Direct Coupling Analysis")
    parser.add_argument("--version", action="version", version=version("annadca"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "sample"):
        subparsers.add_parser(command, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command == "train":
        from annadca.scripts.train import main as run
    else:
        from annadca.scripts.sample import main as run
    run(remaining)


if __name__ == "__main__":
    main()
