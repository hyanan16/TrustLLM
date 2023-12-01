import argparse
from robustness import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default=None)
    parser.add_argument('--evaltype', type=str, default=None)
    parser.add_argument('--savepath', type=str, default=None)
    args = parser.parse_args()
    if args.evaltype not in ['advglue', 'advinstruction_res', 'advinstruction_sim', 'ood_detection']:
        raise ValueError('Invalid evaluation type!')
    if args.evaltype == 'advglue':
        advglue_eval(args.folderpath, args.savepath)
    elif args.evaltype == 'advinstruction_res':
        adv_instruction.gen_advinstruction_res(args.folderpath, args.savepath)
    elif args.evaltype == 'advinstruction_sim':
        get_embedding.embedding_dir(args.folderpath)
    elif args.evaltype == 'ood_detection':
        ood_detection(args.folderpath, args.savepath)
