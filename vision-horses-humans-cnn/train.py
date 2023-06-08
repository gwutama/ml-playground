import common
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Common constants
TRAINING_DIR = 'data/images/training'
VALIDATION_DIR = 'data/images/validation'
NUM_EPOCHS = 15


def download_dataset():
    common.download_extract_dataset(url='https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip',
                                    extract_dir=TRAINING_DIR)
    common.download_extract_dataset(url='https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip',
                                    extract_dir=VALIDATION_DIR)


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

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    download_dataset()
    train_generator, validation_generator = prepare_dataset(TRAINING_DIR, VALIDATION_DIR)
    model = build_compile_model()
    model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=validation_generator)
    model.summary()
    model.save('model.tf')