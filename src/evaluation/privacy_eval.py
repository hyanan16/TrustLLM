import argparse
from robustness import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    args = parser.parse_args()
    if args.evaltype not in ['download', 'openended']:
        raise ValueError('Invalid evaluation type!')
    elif args.evaltype == 'download':
        advglue_eval(args.folderpath, args.savepath)
    elif args.evaltype == 'openended':
        similarity_embedding(args.folderpath, args.savepath)