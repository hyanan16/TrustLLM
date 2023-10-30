import argparse
from robustness import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default=None)
    parser.add_argument('--evaltype', type=str, default=None)
    parser.add_argument('--savepath', type=str, default=None)
    parser.add_argument('--embeddingpath', type=str, default=None)
    args = parser.parse_args()
    if args.evaltype not in ['advglue', 'advinstruction_res', 'advinstruction_sim']:
        raise ValueError('Invalid evaluation type!')
    if args.evaltype == 'advglue':
        advglue_eval(args.folderpath, args.savepath)
    elif args.evaltype == 'advinstruction_res':
        similarity_embedding.gen_advinstruction_res(args.folderpath, args.savepath)
    elif args.evaltype == 'advinstruction_sim':
        similarity_embedding.calculate_and_save_similarities(args.folderpath, args.embeddingpath, args.savepath)
