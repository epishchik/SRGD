from argparse import ArgumentParser

import netron

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="input model file in onnx format",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host address")
    parser.add_argument("--port", type=int, default=12651, help="port number")
    args = parser.parse_args()

    netron.start(file=args.file, address=(args.host, args.port), browse=True)
