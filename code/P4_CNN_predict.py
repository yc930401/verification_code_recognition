from P2_data_preparation import get_data, get_charset, get_label
from P3_CNN_train import convert_labels, calculate_acc
from keras.models import load_model
import numpy as np

chars = get_charset()
test_size = 10
num_digits = 5

if __name__ == '__main__':
    x_test, y_test = get_data(test_size)
    y_test = convert_labels(y_test)
    model = load_model('model_CNN.h5')
    y_pred = model.predict(x_test)

    for i in range(test_size):
        actual_labels = []
        predicted_labels = []

        for j in range(num_digits):
            actual_labels.append(np.argmax(y_test[j][i]))
            predicted_labels.append(np.argmax(y_pred[j][i]))

        print("Actual labels: {}".format(get_label(actual_labels)))
        print("Predicted labels: {}\n".format(get_label(predicted_labels)))




