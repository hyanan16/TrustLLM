import argparse
from privacy import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    parser.add_argument('--datatype', type=str, default='normal')
    args = parser.parse_args()
    if args.evaltype not in ['awareness', 'leakage', 'longformer']:
        raise ValueError('Invalid evaluation type')
    if args.evaltype == 'awareness':
        privacy_awareness.get_privacy_awareness_res(args.folderpath, args.savepath, args.datatype)
    elif args.evaltype == 'leakage':
        privacy_leakage.run_privacy_leakage(args.folderpath, args.savepath)
    elif args.evaltype == 'longformer':
        longformer.run_privacy_longformer(args.folderpath, args.savepath)