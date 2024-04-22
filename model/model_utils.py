import keras
from keras import backend as K
import numpy as np
from phe import paillier  # Homomorphic Encryption library
import pickle
import matplotlib.pyplot as plt


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

def train_model(model, X_train, y_train,epochs=5, batch_size=16):
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

def aggregate_models(models):
    weights = [model.get_weights() for model in models]
    average_weights = []

    # Compute the mean in smaller batches
    batch_size = 10  # Adjust this value as needed
    for i in range(0, len(weights), batch_size):
        batch_weights = weights[i:i+batch_size]
        average_batch_weights = [np.mean([weight[j] for weight in batch_weights], axis=0) for j in range(len(batch_weights[0]))]
        average_weights.append(average_batch_weights)

    # Compute the overall mean
    average_weights = [np.mean([weight[i] for weight in average_weights], axis=0) for i in range(len(average_weights[0]))]

    aggregated_model = keras.models.clone_model(models[0])
    aggregated_model.set_weights(average_weights)
    return aggregated_model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

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


    global_model = initialize_model()
    all_histories = [[] for _ in range(len(data_splits))]
    global_model_history = []
    no_of_communication_rounds = 3
    for round in range(no_of_communication_rounds):
        print(f'Starting round {round+1}...')
        models = []
        for i, (X_split, y_split) in enumerate(data_splits):
            model = keras.models.clone_model(global_model)
            model.set_weights(global_model.get_weights())
            m, history = train_model(model, X_split, y_split, epochs=1)
            models.append(m)
            all_histories[i].append(history)

        global_model = aggregate_models(models)    
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
    plt.ylim(0, 1)
    plt.legend()
    plt.show()