import keras
from keras import backend as K
import numpy as np
from phe import paillier  # Homomorphic Encryption library
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data(filepath='local_data1.pkl'):
    with open(filepath, 'rb') as f:
        (X_train, y_train) = pickle.load(f)
    return X_train, y_train

def ewc_loss(star_vars, lambd):
    def loss(y_true, y_pred):
        c = []
        for v in range(len(model.trainable_weights)):
            c.append(K.sum(K.square(model.trainable_weights[v] - K.constant(star_vars[v]))))
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

def initialize_simple_model(star_vars=None, lambd=4000):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=ewc_loss(star_vars, lambd) if star_vars is not None else 'categorical_crossentropy')
    return model

def train_model(model, X_train, y_train, epochs=5, batch_size=16):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model, history

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

def aggregate_models(models, star_vars, lambd=4000, batch_size=16):
    weights = [model.get_weights() for model in models]
    average_weights = []

    for i in range(0, len(weights), batch_size):
        batch_weights = weights[i:i+batch_size]
        average_batch_weights = [np.mean([weight[j] for weight in batch_weights], axis=0) for j in range(len(batch_weights[0]))]
        average_weights.append(average_batch_weights)

    # Compute the overall mean
    average_weights = [np.mean([weight[i] for weight in average_weights], axis=0) for i in range(len(average_weights[0]))]

    aggregated_model = keras.models.clone_model(models[0])
    aggregated_model.set_weights(average_weights)

    # Apply EWC loss
    aggregated_model.compile(optimizer='adam', loss=ewc_loss(star_vars, lambd), metrics=['accuracy'])

    return aggregated_model

if __name__ == '__main__':

    X_test, y_test = load_data('server_data/local_data.pkl')

    data_splits = []
    for i in range(1, 6):
        X_train, y_train = load_data(f'contributor{i}/local_data.pkl')
        data_splits.append((X_train, y_train))

    global_model = initialize_model()

    all_histories = [[] for _ in range(len(data_splits))]
    global_model_history = []
    no_of_communication_rounds = 10

    for round in range(no_of_communication_rounds):
        print(f'Starting round {round+1}...')
        models = []
        for i, (X_split, y_split) in enumerate(data_splits):
            # this code is run on each client server
            model = keras.models.clone_model(global_model)
            model.set_weights(global_model.get_weights())
            m, history = train_model(model, X_split, y_split, epochs=1,batch_size=32)
            models.append(m)
            all_histories[i].append(history)
        if round == 0:
            star_vars = [var.numpy() for var in global_model.weights]
        # then central server aggregates the models
        global_model = aggregate_models(models, star_vars, lambd=4000, batch_size=32) 
        loss, accuracy = evaluate_model(global_model, X_test, y_test)
        global_model_history.append((loss, accuracy))

    # Plot accuracy for each client
    for i in range(len(data_splits)):
        client_histories = [h.history['accuracy'] for h in all_histories[i]]
        plt.plot(client_histories, label=f'Client {i+1}')

    # Plot accuracy for aggregated model
    global_model_histories = [h[1] for h in global_model_history]
    plt.plot(global_model_histories, label='Aggregated model')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # Set the x-axis to only display integer values
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.ylim(.77, 1)
    plt.legend()
    plt.show()