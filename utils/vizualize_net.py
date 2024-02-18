from argparse import ArgumentParser

import netron

if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO добавить help
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12651)
    args = parser.parse_args()

    netron.start(file=args.file, address=(args.host, args.port), browse=True)
