import sys

COMMANDS = ("train", "sample")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 1 or argv[0] not in COMMANDS:
        print(f"Usage: annadca {{{','.join(COMMANDS)}}} [options]. Use 'annadca <command> -h' for the options of each command.")
        sys.exit(1)
    command, args = argv[0], argv[1:]
    if command == "train":
        from annadca.scripts.train import main as command_main
    else:
        from annadca.scripts.sample import main as command_main
    command_main(args)


if __name__ == "__main__":
    main()
