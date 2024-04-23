import keras
import numpy as np
import os
import pickle

def distribute_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    directory = 'server_data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the test data to a file
    with open(f'{directory}/local_data.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)
    
    label_sets = [
        {1,3,4,7},
        {0,5,6,8},
        {1,2,4,9},
        {0,2,3,5},
        {6,7,8,9}
    ]

    data_splits = [[] for _ in range(len(label_sets))]

    for label in range(10):
        label_indices = np.where(np.argmax(y_train, axis=-1) == label)[0].astype(int)
    
        first_match = True
        half = len(label_indices) // 2
        
        for i, label_set in enumerate(label_sets):
            if label in label_set:
                if first_match:
                    # first time we match a label in the set, use the first half of the data
                    X_split, y_split = X_train[label_indices[:half]], y_train[label_indices[:half]]
                    first_match = False
                else:
                    # second time we match a label in the set, use the second half of the data
                    X_split, y_split = X_train[label_indices[half:]], y_train[label_indices[half:]]
                data_splits[i].append((X_split, y_split))

    for i, split in enumerate(data_splits):
        X_splits, y_splits = zip(*split)
        X_splits = np.concatenate(X_splits)
        y_splits = np.concatenate(y_splits)
        # shuffle the data
        indices = np.arange(len(X_splits))
        np.random.shuffle(indices)
        data_splits[i] = (X_splits[indices], y_splits[indices])

    for i, (X_split, y_split) in enumerate(data_splits):
        with open(f'{directory}/local_data_{i}.pkl', 'wb') as f:
            pickle.dump((X_split, y_split), f)

if __name__ == '__main__':
    distribute_data()