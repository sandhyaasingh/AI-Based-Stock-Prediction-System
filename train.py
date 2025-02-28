import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from pre import preprocess_data

def create_dataset(data, time_step=1):
    """
    Create dataset for LSTM model training.

    Parameters:
    data (np.array): Scaled data.
    time_step (int): Number of time steps for each sample.

    Returns:
    np.array: Features (X) and target (Y) arrays.
    """
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def add_features(df):
    """
    Add moving averages and volatility to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with added features.
    """
    # Ensure the correct column name is used
    if 'y' not in df.columns:
        raise KeyError("Column 'y' not found in DataFrame")

    df['MA_10'] = df['y'].rolling(window=10).mean()
    df['MA_20'] = df['y'].rolling(window=20).mean()
    df['Volatility'] = df['y'].rolling(window=10).std()
    df.bfill(inplace=True) 
    return df

def train_model(file_path, model_path='stock_model.h5', retrain= True):
    """
    Train LSTM model on stock price data.

    Parameters:
    file_path (str): Path to the CSV file.
    model_path (str): Path to save/load the model.
    retrain (bool): Whether to retrain the model.

    Returns:
    tuple: Trained model, training and test datasets, and scaler.
    """
    # Preprocess data and add features
    scaled_df, scaler = preprocess_data(file_path)
    scaled_df = add_features(scaled_df)
    scaled_data = scaled_df.values

    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    # Check if the model already exists and retrain flag is False
    if os.path.exists(model_path) and not retrain:
        model = load_model(model_path)
        print("Model loaded from disk.")
    else:
        # Build the LSTM model with Dropout layers and L2 regularization
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(Dropout(0.3))  # First LSTM layer with 0.3 dropout
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))  # Second LSTM layer with 0.2 dropout
        model.add(Dense(25, kernel_regularizer=l2(0.01)))  # Add L2 regularization
        model.add(Dense(1, kernel_regularizer=l2(0.01)))  # Add L2 regularization

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, Y_train, batch_size=1, epochs=20)

        # Save the model to disk
        model.save(model_path)
        print("Model saved to disk.")

    return model, X_train, X_test, Y_train, Y_test, scaler

if __name__ == "__main__":
    model, X_train, X_test, Y_train, Y_test, scaler = train_model('stock_price.csv', retrain= False)