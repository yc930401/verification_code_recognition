import os.path
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from P2_data_preparation import get_data, get_charset

index_to_char, char_to_index = get_charset()
train_data_size = 1000
test_data_size = 100
batch_size = 100
num_digits = 5
num_classes = len(index_to_char)
epochs = 30


def convert_labels(labels):

    # Declare output ndarrays
    dig0_arr = np.ndarray(shape=(len(labels), num_classes))
    dig1_arr = np.ndarray(shape=(len(labels), num_classes))
    dig2_arr = np.ndarray(shape=(len(labels), num_classes))
    dig3_arr = np.ndarray(shape=(len(labels), num_classes))
    dig4_arr = np.ndarray(shape=(len(labels), num_classes))

    for index, label in enumerate(labels):
        # one hot encoding
        dig0_arr[index, :] = np_utils.to_categorical(label[0], num_classes)
        dig1_arr[index, :] = np_utils.to_categorical(label[1], num_classes)
        dig2_arr[index, :] = np_utils.to_categorical(label[2], num_classes)
        dig3_arr[index, :] = np_utils.to_categorical(label[3], num_classes)
        dig4_arr[index, :] = np_utils.to_categorical(label[4], num_classes)

    return [dig0_arr, dig1_arr, dig2_arr, dig3_arr, dig4_arr]


def CNN_model():

    # image input dimensions
    img_rows = 50
    img_cols = 150
    img_channels = 1

    # defining the input
    inputs = Input(shape=(img_rows, img_cols, img_channels))

    # Model taken from keras example. Worked well for a digit, dunno for multiple
    cov = Conv2D(32, (3, 3), padding="same")(inputs)
    cov = Activation('relu')(cov)
    cov = Conv2D(32, (3, 3))(cov)
    cov = Activation('relu')(cov)
    cov = MaxPooling2D(pool_size=(2, 2))(cov)
    cov = Dropout(0.5)(cov)
    cov_out = Flatten()(cov)

    cov2 = Dense(128, activation='relu')(cov_out)
    cov2 = Dropout(0.5)(cov2)

    # Prediction layers
    c0 = Dense(num_classes, activation='softmax')(cov2)
    c1 = Dense(num_classes, activation='softmax')(cov2)
    c2 = Dense(num_classes, activation='softmax')(cov2)
    c3 = Dense(num_classes, activation='softmax')(cov2)
    c4 = Dense(num_classes, activation='softmax')(cov2)

    # Defining the model
    model = Model(inputs=inputs, outputs=[c0, c1, c2, c3, c4])

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_CNN(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_test, y_test))

    model.save('model_CNN.h5')
    return model


def calculate_acc(test_data_size, predictions, real_labels):
    print(np.shape(predictions))
    individual_counter = 0
    global_sequence_counter = 0
    for i in range(test_data_size):
        sequence_counter = 0
        for j in range(num_digits):
            if np.argmax(predictions[j][i]) == np.argmax(real_labels[j][i]):
                individual_counter += 1
                sequence_counter += 1

        if sequence_counter == num_digits:
            global_sequence_counter += 1
    print('individual_counter: ', individual_counter)
    print('total digits: ', test_data_size * num_digits)
    ind_accuracy = individual_counter / (test_data_size * num_digits)
    global_accuracy = global_sequence_counter / test_data_size

    return ind_accuracy, global_accuracy


if __name__ == '__main__':

    if os.path.exists('model_CNN.h5'):
        model = load_model('model_CNN.h5')
    else:
        model = CNN_model()
    count = 0
    while count < 1000:
        x_train, y_train = get_data(train_data_size)
        x_test, y_test = get_data(test_data_size)
        y_train = convert_labels(y_train)
        y_test = convert_labels(y_test)
        print(np.shape(x_train), np.shape(y_test))

        model = train_CNN(model, x_train, y_train, x_test, y_test)
        y_pred = model.predict(x_test)
        ind_acc, glob_acc = calculate_acc(test_data_size, y_pred, y_test)
        count += 1 #glob_acc

        print("The individual accuracy is {} %".format(ind_acc * 100))
        print("The sequence prediction accuracy is {} %".format(glob_acc * 100))
