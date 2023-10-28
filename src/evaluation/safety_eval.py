import argparse
from safety import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    args = parser.parse_args()
    if args.evaltype not in ['all', 'jailbreak_model', 'jailbreak_type', 'toxicity', 'misuse']:
        raise ValueError('Invalid evaluation type!')
