import argparse
from truthfulness import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaltype", type=str, default=None, help="which part to evaluate"
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="../../results",
        help="where to save evaluation results",
    )
    args = parser.parse_args()
    if args.evaltype not in [
        "internal",
        "external",
        "hallucination",
        "sycophancy",
        "advfactuality",
    ]:
        raise ValueError("Invalid evaluation type!")
    if args.evaltype == "internal":
        internal.run(args.savepath)
    elif args.evaltype == "external":
        external.run(args.savepath)
    elif args.evaltype == "hallucination":
        hallucination.run(args.savepath)
    elif args.evaltype == "sycophancy":
        sycophancy.run(args.savepath)
    elif args.evaltype == "advfactuality":
        advfactuality.run(args.savepath)
