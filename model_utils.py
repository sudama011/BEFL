import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path='diabetes.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def initialize_model():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10):
    model.fit(X_train, y_train, epochs=epochs)
    return model

def evaluate_model(model, X_test, y_test):
    model.compile(optimizer='adam', loss='mean_squared_error')
    mse = model.evaluate(X_test, y_test)
    return mse

df = load_data('diabetes.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)
X_train, X_test = standardize_data(X_train, X_test)
