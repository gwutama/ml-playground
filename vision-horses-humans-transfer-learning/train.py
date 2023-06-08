import common
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Common constants
TRAINING_DIR = 'data/images/training'
VALIDATION_DIR = 'data/images/validation'
NUM_EPOCHS = 20


def download_dataset():
    common.download_extract_dataset(url='https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip',
                                    extract_dir=TRAINING_DIR)
    common.download_extract_dataset(url='https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip',
                                    extract_dir=VALIDATION_DIR)


def prepare_dataset(training_dir, validation_dir):
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        batch_size=20,
        target_size=(150, 150),
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=20,
        target_size=(150, 150),
        class_mode='binary')

    return (train_generator, validation_generator)


def build_compile_model():
    weights_file = common.download_file('https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)
    pre_trained_model.load_weights(weights_file)

    # Freeze the entire network from retraining
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Add our dense layers underneath the pre-trained model
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)

    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)

    # Add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(pre_trained_model.input, x)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    download_dataset()
    train_generator, validation_generator = prepare_dataset(TRAINING_DIR, VALIDATION_DIR)
    model = build_compile_model()
    model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=validation_generator, verbose=1)
    model.summary()
    model.save('model.tf')