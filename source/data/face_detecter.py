# coding:utf-8
import os, sys, time
import argparse, glob
import pandas as pd
import face_recognition
from PIL import Image

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
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(input_path)

    H, W, _ = image.shape

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    # Extract when just one face is detected
    if len(face_locations)==1:
        #for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_locations[0]
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[max(top-MARGIN,0):min(H-1, bottom+MARGIN), max(left-MARGIN,0):min(W-1, right+MARGIN)]
        pil_image = Image.fromarray(face_image)
        pil_image = pil_image.resize((224,224), resample=Image.BICUBIC)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        pil_image.save(output_path, 'JPEG', quality=100, optimize=True)

if __name__ == '__main__':
    main()
