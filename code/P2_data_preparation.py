import numpy as np
import P1_code_generation as generate

def get_charset():
    #chars = [chr(i) for i in range(65,91)] + [str(i) for i in range(0,10)]
    chars = [str(i) for i in range(0,10)]
    index_to_char = {i:char for i,char in enumerate(chars)}
    char_to_index = {char:i for i,char in enumerate(chars)}
    return index_to_char, char_to_index


def get_data(batch_size=100):
    x_data = []
    y_data = []
    index_to_char, char_to_index = get_charset()
    for i in range(batch_size):
        x, y = generate.get_data()
        x = np.array(x.convert('L'))
        #y = np.array([[1 if item == char else 0 for item in y] for char in chars]).T
        y = [char_to_index[i] for i in y]
        np.set_printoptions(threshold=np.inf)
        x_data.append(x)
        y_data.append(y)

    x_data = np.array(x_data).reshape(batch_size, 50, 150, 1).astype('float32')/255
    y_data = np.array(y_data)
    return x_data, y_data


def get_label(x_data):
    index_to_char, char_to_index = get_charset()
    results = [index_to_char[i] for i in x_data]
    return results
