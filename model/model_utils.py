import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from phe import paillier  # Homomorphic Encryption library
from sklearn.model_selection import train_test_split
from tensorflow import reduce_mean
import pickle


def load_data(filepath='local_data1.pkl'):
    with open(filepath, 'rb') as f:
        X_train, y_train = pickle.load(f)
    return X_train, y_train

def ewc_loss(star_vars, lambd):
    def loss(y_true, y_pred):
        c = []
        for v in range(len(model.trainable_variables)):
            c.append(K.sum(K.square(model.trainable_variables[v] - star_vars[v])))
        return K.categorical_crossentropy(y_true, y_pred) + lambd * K.sum(c)
    return loss

def initialize_model(star_vars=None, lambd=4000):  # lambd is the EWC regularization strength
    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu'),
        keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=ewc_loss(star_vars, lambd) if star_vars is not None else 'categorical_crossentropy')
    return model

def initialize_simple_model():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def train_model(model, X_train, y_train, epochs=5):
    model.fit(X_train, y_train, epochs=epochs)
    return model

def encrypt_model_params(model, public_key):
    encrypted_weights = []
    for layer in model.layers:
        encrypted_layer_weights = []
        for weight_array in layer.get_weights():
            encrypted_weights_array = np.array([[public_key.encrypt(float(w)) for w in row] for row in weight_array])
            encrypted_layer_weights.append(encrypted_weights_array)
        encrypted_weights.append(encrypted_layer_weights)
    return encrypted_weights

def decrypt_model_params(encrypted_weights, private_key):
    decrypted_weights = []
    for layer in encrypted_weights:
        decrypted_layer_weights = []
        for weight_array in layer:
            decrypted_weights_array = np.array([[private_key.decrypt(w) for w in row] for row in weight_array])
            decrypted_layer_weights.append(decrypted_weights_array)
        decrypted_weights.append(decrypted_layer_weights)
    return decrypted_weights

def evaluate_model(model, X_test, y_test):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy

def aggregate_models(models):
    weights = [model.get_weights() for model in models]
    average_weights = [reduce_mean([weight[i] for weight in weights], axis=0) for i in range(len(weights[0]))]
    aggregated_model = keras.models.clone_model(models[0])
    aggregated_model.set_weights(average_weights)
    return aggregated_model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    label_sets = [
        {1,3,4,7},
        {0,5,6,8},
        {1,2,4,9},
        {0,2,3,5},
        {6,7,8,9}
    ]

    used_indices = set()

    data_splits = []
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
        
        data_splits.append((X_split, y_split))

    models = []
    for X_split, y_split in data_splits:
        model = initialize_model()
        m = train_model(model, X_split, y_split)
        models.append(m)

    aggregated_model = aggregate_models(models)    
    loss, accuracy = evaluate_model(aggregated_model, X_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')