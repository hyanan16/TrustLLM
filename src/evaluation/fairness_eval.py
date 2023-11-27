import argparse
from fairness import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, default='./')
    parser.add_argument('--evaltype', type=str, default='all')
    parser.add_argument('--savepath', type=str, default='./')
    args = parser.parse_args()
    if args.evaltype not in ['all', 'agreement', 'stereroset', 'preference', 'disparagement']:
        raise ValueError('Invalid evaluation type!')
    if args.evaltype == 'agreement':
        stereotype_agreement.run_stereotype_agreement_eval(args.folderpath, args.savepath)
    elif args.evaltype == 'stereoset':
        stereotype_stereroset.run_stereotype_classification_eval(args.folderpath, args.savepath)
    elif args.evaltype == 'preference':
        preference.calculate_preference_RtA_plain(args.folderpath, args.savepath)
        preference.calculate_preference_RtA_force_selection(args.folderpath, args.savepath)
    elif args.evaltype == 'disparagement':
        disparagement.chi_square_csv_save(args.folderpath, args.savepath)
