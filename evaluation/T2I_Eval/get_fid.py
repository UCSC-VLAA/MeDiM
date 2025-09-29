from cleanfid import fid
import os
import argparse
os.environ['NCCL_DEBUG']="None"


def main(args):
    score = fid.compute_fid(args.src_dir, args.gen_dir)
    print(score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--src_dir', type=str, help='path to MJHQ-30K images')
    parser.add_argument('--gen_dir', type=str, help='path to generated 30K images')

    args = parser.parse_args()
    main(args)



