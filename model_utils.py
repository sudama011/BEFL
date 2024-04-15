import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from phe import paillier  # Homomorphic Encryption library

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255
    X_test = X_test.reshape(-1, 784).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def initialize_model(star_vars=None, lambd=4000):  # lambd is the EWC regularization strength
    model = keras.models.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=ewc_loss(star_vars, lambd) if star_vars is not None else 'categorical_crossentropy')
    return model

def ewc_loss(star_vars, lambd):
    def loss(y_true, y_pred):
        c = []
        for v in range(len(model.trainable_variables)):
            c.append(K.sum(K.square(model.trainable_variables[v] - star_vars[v])))
        return K.categorical_crossentropy(y_true, y_pred) + lambd * K.sum(c)
    return loss

def train_model(model, X_train, y_train, epochs=10):
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

# X_train, X_test, y_train, y_test = load_data()
# model = initialize_model()
# model = train_model(model, X_train, y_train)
# public_key, private_key = paillier.generate_paillier_keypair()  # Generate keys for Homomorphic Encryption
# # encrypted_weights = encrypt_model_params(model, public_key)
# # decrypted_weights = decrypt_model_params(encrypted_weights, private_key)
# # for layer, weights in zip(model.layers, decrypted_weights):
# #     layer.set_weights(weights)
# loss, accuracy = evaluate_model(model, X_test, y_test)
# print(f'Test loss: {loss}, Test accuracy: {accuracy}')
