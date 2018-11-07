#coding:utf-8
import os, sys, time
import argparse
import pandas as pd
from distutils.dir_util import copy_tree

def main():
    parser = argparse.ArgumentParser(description='change orthographic variants')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--input_dir', '-i', default='../../data/raw/photos',
                        help='input directory')
    parser.add_argument('--annotation', default='../../data/raw/annotation.csv',
                        help='ground truth csv')
    parser.add_argument('--output_dir', '-o', default='../../data/interim',
                        help='output directory')
    args = parser.parse_args()

    output_dir = os.path.join(args.base, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(os.path.join(args.base, args.annotation))

    new_df = df.replace(r'miss*', '', regex=True)
    new_df = new_df.replace(r'todai', 'tokyo', regex=True)

    print('----- Copy start -----')
    input_dir = os.path.join(args.base, args.input_dir)
    cols = ['path', 'college', 'year', 'entry_num', 'label']
    error_df = pd.DataFrame(columns=cols)
    for path, new_path in zip(df['path'], new_df['path']):
        input_path = os.path.join(input_dir, path)
        output_path = os.path.join(output_dir, new_path)
        try:
            copy_tree(input_path, output_path)
        except:
            print("Copy error: {}".format(path))
            error_df = error_df.append( new_df[new_df['path']==new_path], ignore_index=True)
            new_df = new_df[new_df['path']!=new_path]

    new_df.to_csv('{}/annotation.csv'.format(output_dir), index=False, encoding='utf-8', mode='w')
    error_df.to_csv('{}/copy_error.csv'.format(output_dir), index=False, encoding='utf-8', mode='w')


if __name__ == '__main__':
    main()
