import argparse
from safety import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    args = parser.parse_args()
    if args.evaltype not in ['jailbreak_model', 'jailbreak_type', 'toxicity', 'misuse']:
        raise ValueError('Invalid evaluation type!')
    elif args.evaltype == 'jailbreak_model':
        jailbreak_model_stat(args.folderpath, args.savepath)
    elif args.evaltype == 'jailbreak_type':
        jailbreak_type_stat(args.folderpath, args.savepath)
    elif args.evaltype == 'toxicity':
        toxicity_pespective(args.folderpath, args.savepath)
    elif args.evaltype == 'misuse':
        misuse_stat(args.folderpath, args.savepath)

