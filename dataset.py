# coding:utf-8
import os, sys, time
import numpy as np
import chainer
import argparse, glob
import pandas as pd
import matplotlib.pyplot as plt

class BeautyDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, path, mean_path, augmentation=False):
        """
        @param root: Root directory to retrieve images from.
        @param path: txt path that is included (imagepath label)
        @param mean_path: Path to mean image
        """
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.augmentation = augmentation
        mean = np.load(mean_path)
        self.mean = mean.mean(axis=(1,2))
        
    def __len__(self):
        return len(self.base)

    def transform(self, image):
        if np.random.rand() > 0.5:
            image = image[..., ::-1]
        return image

    def get_example(self, i):
        image, label = self.base[i]
        image = image - self.mean[:, None, None]
        image = image.astype(np.float32)
        if not self.augmentation:
            return image, label

        image = self.transform(image)
        return image, label

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make dataset txt list')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--input_dir', '-i', default='data/preprocessed',
                        help='input directory')
    parser.add_argument('--annotation', default='data/preprocessed/annotation.csv',
                        help='ground truth csv')

    parser.add_argument('--root', '-R', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/preprocessed'),
                        help='base directory path of program files')
    parser.add_argument('--output_dir', '-o', default='data/preprocessed',
                        help='output directory')
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(os.path.join(args.base, args.annotation))

    # Shuffle dataset
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    # Split dataset
    # Train: 60% Validation: 20% Test: 20%
    # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    train, validate, test = np.split(df, [int(0.6*len(df)), int(0.8*len(df))])

    def generate_data_list(root, paths, labels, output_path):
        """
        @param root: Root directory to retrieve images from.
        @param paths: It contains of image path
        @param labels: It is labels corresponding image path
        @param output_path: Output file path
        """
        count = 0
        n_image_list = []
        for basepath, label in zip(paths, labels):
            input_dir = os.path.join(root, basepath)
            fnames = glob.glob('{}/*.jpg'.format(input_dir))

            for fname in fnames:
                filename = os.path.basename(fname)
                n_image_list.append([os.path.join(basepath, filename), int(label)])
                count+=1
                if count%100 ==0:
                    print(count)

        print("Num of examples:{}".format(count))
        n_image_list = np.array(n_image_list, np.str)
        np.savetxt(output_path, n_image_list, fmt="%s")

    def make_mean_img(root, image_list, output_path):
        t = chainer.datasets.LabeledImageDataset(image_list, root)
        tmp, _ = t[0]
        mean = np.zeros(tmp.shape)
        for img, _ in t:
            mean += img
        mean = mean/float(len(t))
        #plt.imshow(mean.transpose(1, 2, 0)/255)
        #plt.show()
        np.save(output_path, mean)

    output_dir = os.path.join(args.base, args.output_dir)
    input_dir = os.path.join(args.base, args.input_dir)

    print('----- Make train list and mean image -----')
    output_path = '{}/train_list.txt'.format(output_dir)
    generate_data_list(input_dir, train['path'], train['label'], output_path)
    make_mean_img(args.root, output_path, '{}/train_mean.npy'.format(output_dir))

    print('----- Make validation list -----')
    output_path = '{}/val_list.txt'.format(output_dir)
    generate_data_list(input_dir, validate['path'], validate['label'], output_path)

    print('----- Make test list -----')
    output_path = '{}/test_list.txt'.format(output_dir)
    generate_data_list(input_dir, test['path'], test['label'], output_path)
