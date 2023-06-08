import common
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Common constants
TRAINING_DIR = 'data/images/training'
VALIDATION_DIR = 'data/images/validation'
TMP_DIR = 'data/tmp'
NUM_EPOCHS = 20


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
        batch_size=100,
        target_size=(150, 150),
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=100,
        target_size=(150, 150),
        class_mode='binary')

    return (train_generator, validation_generator)


def build_compile_model():
    weights_file = common.download('https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
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