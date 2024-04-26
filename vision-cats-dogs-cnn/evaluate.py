#!/usr/bin/env python

import argparse
import numpy as np
import tensorflow as tf
import keras.preprocessing.image as keras_image

if __name__ == '__main__':
    # read all arguments from command line
    # Example: evaluate.py image1.jpg image2.jpg ...
    parser = argparse.ArgumentParser(description='Predict cat or dog')
    parser.add_argument('images', metavar='image', type=str, nargs='+')
    args = parser.parse_args()
    images_filepaths = args.images

    # load the model
    model = tf.keras.models.load_model('model_2504.tf')

    for image_filepath in images_filepaths:
        img = keras_image.load_img(image_filepath, target_size=(150, 150))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)
        if classes[0] == 1.0:
            print(image_filepath + " is a dog")
        else:
            print(image_filepath + " is a cat")
