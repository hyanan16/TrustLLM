import argparse
from ethics import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    args = parser.parse_args()
    if args.evaltype not in ['all', 'explicit', 'implicit', 'emotional']:
        raise ValueError('Invalid evaluation type!')
    elif args.evaltype == 'all':
        pass
    elif args.evaltype == 'explicit':
        explicit_ethics(args.folderpath, args.savepath)
    elif args.evaltype == 'implicit':
        pass
    elif args.evaltype == 'emotional':
        emotion_awareness(args.folderpath, args.savepath)