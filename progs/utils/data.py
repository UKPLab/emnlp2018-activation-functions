import numpy as np

#  Reads data from src
from tflearn.data_utils import pad_sequences


def read_data(src):
    with open(src, 'r') as src_file:
        vecs = []
        sentences = []
        # go to the 0-th Byte in file
        src_file.seek(0)
        # i is the index, line is the data
        for i, line in enumerate(src_file):
            line = line.rstrip()
            str_vec = line.split(' ')
            #str_vec.append(1)
            # convert to floatss
            vector = [float(x) for x in str_vec]
            vecs.append(vector)
    return np.array(vecs)


# Merges / Zips data of two classes
def merge(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i])
        z.append(y[i])

    return np.array(z)


# read data for newsgroups dataset
def read_data_ng(input_file, pad_length):
    vocab = {0}
    data_x = []
    data_y = []
    with open(input_file) as f:
        for line in f:
            label, content = line.split('\t')
            content = [int(v) for v in content.split()]
            vocab.update(content)
            data_x.append(content)
            label = tuple(int(v) for v in label.split())
            data_y.append(label)

    data_x = pad_sequences(data_x, maxlen=pad_length)
    return list(zip(data_y, data_x)), vocab

def load_labels_trec(src):
    options = {
        'ABBR': [1, 0, 0, 0, 0, 0],
        'DESC': [0, 1, 0, 0, 0, 0],
        'ENTY': [0, 0, 1, 0, 0, 0],
        'HUM': [0, 0, 0, 1, 0, 0],
        'LOC': [0, 0, 0, 0, 1, 0],
        'NUM': [0, 0, 0, 0, 0, 1]
    }
    with open(src, 'r') as src_file:
        labels = []
        # go to the 0-th Byte in file
        src_file.seek(0)
        # i is the index, line is the data
        for i, line in enumerate(src_file):
            label = str(line.rstrip())
            labels.append(options[label])

    return np.array(labels)

def load_labels_pe(src):
    options = {
        'O': [1, 0, 0, 0],
        'MajorClaim': [0, 1, 0, 0],
        'Claim': [0, 0, 1, 0],
        'Premise': [0, 0, 0, 1]
    }
    with open(src, 'r') as src_file:
        labels = []
        # go to the 0-th Byte in file
        src_file.seek(0)
        # i is the index, line is the data
        for i, line in enumerate(src_file):
            label = str(line.rstrip())
            labels.append(options[label])

    return np.array(labels)
