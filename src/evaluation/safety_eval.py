import argparse
from safety import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    args = parser.parse_args()
    if args.evaltype not in ['longformer', 'jailbreak_model', 'jailbreak_type', 'toxicity_perspective', 'misuse', 'toxicity_visualization', 'toxicity_metrics']:
        raise ValueError('Invalid evaluation type!')
    elif args.evaltype == 'jailbreak_model':
        jailbreak_model_stat(args.folderpath, args.savepath)
    elif args.evaltype == 'jailbreak_type':
        jailbreak_type_stat(args.folderpath, args.savepath)
    elif args.evaltype == 'toxicity_perspective':
        toxicity_pespective(args.folderpath, args.savepath)
    elif args.evaltype == 'toxicity_visualization':
        toxicity_metrics.visualize_toxicity(args.folderpath, args.savepath)
    elif args.evaltype == 'toxicity_metrics':
        toxicity_metrics.calculate_metrics(args.folderpath, args.savepath)
    elif args.evaltype == 'misuse':
        misuse_stat(args.folderpath, args.savepath)
    elif args.evaltype == 'longformer':
        longformer(args.folderpath, args.savepath)

