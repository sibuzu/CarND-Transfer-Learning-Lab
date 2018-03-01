import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential 
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    n_class = np.max(y_train) + 1
    y_train = to_categorical(y_train, n_class)
    y_val = to_categorical(y_val, n_class)

    return X_train, y_train, X_val, y_val

def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    n_input = X_train.shape[1]
    n_output = y_train.shape[1]
    print(n_input, n_output)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(n_input,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])
    model.summary()

    # TODO: train your model here
    model.fit(X_train, y_train, batch_size=64, epochs=20, 
        validation_data=(X_val, y_val), shuffle=True)
    
    score = model.evaluate(X_val, y_val, batch_size=128)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
