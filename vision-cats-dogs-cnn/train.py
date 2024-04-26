#!/usr/bin/env python3

import common
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Common constants
TRAINING_DIR = 'data/images/training'
VALIDATION_DIR = 'data/images/validation'
TMP_DIR = 'data/tmp'
NUM_EPOCHS = 100
SHAPE_WIDTH = 150
SHAPE_HEIGHT = 150

def download_dataset():
    common.unzip('./data/archive/archive.zip', extract_dir=TMP_DIR)

    cat_source_dir = os.path.join(TMP_DIR, 'PetImages/Cat')
    cat_training_dest_dir = os.path.join(TRAINING_DIR, 'cats')
    cat_validation_dest_dir = os.path.join(VALIDATION_DIR, 'cats')
    common.recreate_dir(cat_training_dest_dir)
    common.recreate_dir(cat_validation_dest_dir)
    common.random_split_dataset(cat_source_dir, cat_training_dest_dir, cat_validation_dest_dir)

    dog_source_dir = os.path.join(TMP_DIR, 'PetImages/Dog')
    dog_training_dest_dir = os.path.join(TRAINING_DIR, 'dogs')
    dog_validation_dest_dir = os.path.join(VALIDATION_DIR, 'dogs')
    common.recreate_dir(dog_training_dest_dir)
    common.recreate_dir(dog_validation_dest_dir)
    common.random_split_dataset(dog_source_dir, dog_training_dest_dir, dog_validation_dest_dir)

    common.recreate_dir(TMP_DIR)


def prepare_dataset(training_dir, validation_dir):
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        batch_size=128,
        target_size=(SHAPE_HEIGHT, SHAPE_WIDTH),
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=128,
        target_size=(SHAPE_HEIGHT, SHAPE_WIDTH),
        class_mode='binary')

    return (train_generator, validation_generator)


def build_compile_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(SHAPE_HEIGHT, SHAPE_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    download_dataset()
    train_generator, validation_generator = prepare_dataset(TRAINING_DIR, VALIDATION_DIR)
    model = build_compile_model()
    model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=validation_generator, verbose=1)
    model.summary()
    model.save('model.tf')