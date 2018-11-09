# coding:utf-8
import os, sys, time
import argparse, glob
import pandas as pd
import cv2

def main():

    parser = argparse.ArgumentParser(description='Detect face')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--input_dir', '-i', default='../../data/interim',
                        help='input directory')
    parser.add_argument('--annotation', default='../../data/interim/annotation.csv',
                        help='ground truth csv')
    parser.add_argument('--output_dir', '-o', default='../../data/preprocessed',
                        help='output directory')
    parser.add_argument('--cascade_path', default='../../data/external/haarcascade_frontalface_alt2.xml',
                        help='cascade path (haarcascade_frontalface_alt2.xml or haarcascade_mcs_upperbody.xml) ')
    args = parser.parse_args()

    output_dir = os.path.join(args.base, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(os.path.join(args.base, args.annotation))

    print('----- Start detection ------')
    for path in df['path']:
        input_path = os.path.join(args.base, args.input_dir, path)
        fnames = glob.glob('{}/*.jpg'.format(input_path))

        print('  Case: {}'.format(path))
        counter = 0
        for fname in fnames:
            result_dir = os.path.join(args.base, args.output_dir, path)
            output_path = os.path.join(result_dir, os.path.basename(fname))
            try:
                detect_faces(fname, output_path, os.path.join(args.base, args.cascade_path))
                counter += 1
            except:
                continue

        if counter == 0:
            df = df[df['path']!=path]

    df.to_csv('{}/annotation.csv'.format(output_dir), index=False, encoding='utf-8', mode='w')


def detect_faces(input_path, output_path, cascade_path):
    """
    @param: input_path
                input image path
    @param: output_path
                output image path
    @param: cascade_path
    """

    MARGIN = 30
    cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray_image.shape
    faces = cascade.detectMultiScale(gray_image)
    # Extract when just one face is detected
    if (len(faces) == 1):
        (x, y, w, h) = faces[0]
        image = image[max(y-MARGIN,0):min(H-1, y+h+MARGIN), max(x-MARGIN,0):min(W-1, x+w+MARGIN)]
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        cv2.imwrite(output_path, image)
        #cv2.imshow('image', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print('     I cant detect face:')
        print('         {}'.format(os.path.basename(input_path)))


if __name__ == '__main__':
    main()
