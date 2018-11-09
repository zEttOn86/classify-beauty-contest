# coding:utf-8
import os, sys, time, random
import numpy as np
import argparse, yaml, shutil
import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

from model import MLP
from dataset import BeautyDataset

def main():
    parser = argparse.ArgumentParser(description='Classify beauty contest')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--model', '-m', default='result/Classifier_1593.npz',
                        help='Load model data(snapshot)')
    parser.add_argument('--output_dir', '-o', default='result/inference',
                        help='output directory')

    parser.add_argument('--root', '-R', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/preprocessed'),
                        help='Root directory path of input image')
    parser.add_argument('--test_list', default='test_list.txt',
                        help='test list')
    parser.add_argument('--mean_img', default='train_mean.npy',
                        help='mean image of training list')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('')
    output_dir = os.path.join(args.base, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = L.Classifier(MLP(args.unit, 3))
    model_path = os.path.join(args.base, args.model)
    chainer.serializers.load_npz(model_path, model)
    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
        chainer.global_config.autotune = True
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    xp = model.xp
    path = os.path.join(args.root, args.test_list)
    mean_path = os.path.join(args.root, args.mean_img)
    val = BeautyDataset(root=args.root, path=path, mean_path=mean_path, augmentation=False)

    pred_list =[]
    gt_list = []
    for i, (img, label) in enumerate(val):
        print('Case: {}'.format(val.base._pairs[i][0]))
        xp_img = chainer.Variable(xp.array(img, dtype=xp.float32))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = F.softmax(model.predictor(xp_img[None,...]))

        pred = int(y.array.argmax())
        probability = [float(i) for i in y.array[0]]
        print('   pred: {}, label: {}'.format(pred, label))

        pred_list.append(pred)
        gt_list.append(label)
        if len(pred_list)%10 ==0:
            img += val.mean[:, None, None]
            img = np.asarray(np.clip(img, 0.0, 255.0), dtype=np.uint8)
            plt.figure()
            plt.imshow(img.transpose(1, 2, 0))
            filename = os.path.splitext(val.base._pairs[i][0])[0].replace('/','-').replace('\\', '-')
            plt.title('{} \n pred: {}, label: {}, \n 0: {:.4f}%, 1: {:.4f}%, 2: {:.4f}%'.format(filename, pred, label, probability[0], probability[1], probability[2]))
            output_path = '{}/{}.png'.format(output_dir, filename)
            plt.savefig(output_path)
            plt.clf()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(gt_list, pred_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['First place', 'Second place', 'Else']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Confusion matrix')
    #plt.show()
    output_path = '{}/confusion-matrix.png'.format(output_dir)
    plt.savefig(output_path)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == '__main__':
    main()
