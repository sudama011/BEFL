import pickle
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import os

def distribute_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

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

    used_indices = set()

    for i in range(len(label_sets)):
        # Get the indices of the samples that have labels in the current label set
        indices = np.isin(np.argmax(y_train, axis=1), list(label_sets[i]))
        indices = np.array([index for index in np.where(indices)[0] if index not in used_indices])
        
        # Split the indices into two halves
        half = len(indices) // 2
        if i % 2 == 0:
            indices = indices[:half]
        else:
            indices = indices[half:]
        
        # Add the used indices to the set of used indices
        used_indices.update(indices)

        X_split = X_train[indices]
        y_split = y_train[indices]

        directory = f'contributor{i+1}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the data split to a file
        with open(f'{directory}/local_data.pkl', 'wb') as f:
            pickle.dump((X_split, y_split), f)

if __name__ == '__main__':
    distribute_data()