import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        128, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def train_model(train_set):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_set)

    # Create the training data set
    x_train = []
    y_train = []

    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True,
                                   input_shape=(x_train.shape[1], 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=128)

    return model, scaler


def test_model(test_set, model, scaler):
    # Scale the data
    scaled_data = scaler.transform(test_set)

    # Create the testing data set
    x_test = []
    y_test = []

    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i - 60:i, :])
        y_test.append(scaled_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions on the testing data
    predictions = model.predict(x_test)

    # Reshape the predictions array to match the shape of the original array
    predictions = predictions.reshape(-1, 1)

    # Inverse transform the predictions to get the actual price predictions
    predictions = scaler.inverse_transform(predictions)

    return predictions
