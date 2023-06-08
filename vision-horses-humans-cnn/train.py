import shutil
import os
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_extract_dataset(url, archive_dir, extract_dir, force_download=False, force_delete=True):
    os.makedirs(archive_dir, exist_ok=True)

    # Delete extract_dir if force_delete is True
    if force_delete and os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    # Download file if force_download is True or file does not exist
    # Get filename from url
    filename = os.path.join(archive_dir, url.split('/')[-1])

    if force_download or not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)

    # extract zip file
    if filename.endswith('.zip'):
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(extract_dir)
        zip_ref.close()
    else:
        print('Not a zip file. Skipping %s...'.format(filename))


def prepare_dataset(training_dir, validation_dir):
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        class_mode='binary')

    return (train_generator, validation_generator)


def build_compile_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
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
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    download_extract_dataset(url='https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip',
                             archive_dir='data/archive',
                             extract_dir='data/images/training',
                             force_delete=True)
    download_extract_dataset(url='https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip',
                             archive_dir='data/archive',
                             extract_dir='data/images/validation',
                             force_delete=True)
    train_generator, validation_generator = prepare_dataset('data/images/training', 'data/images/validation')

    model = build_compile_model()
    model.fit(train_generator, epochs=15)
    model.summary()
    model.save('model.tf')