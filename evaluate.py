import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train import train_model

def evaluate_model(file_path):
    """
    Evaluate the trained LSTM model on stock price data.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    None
    """
    model, X_train, X_test, Y_train, Y_test, scaler = train_model(file_path)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Calculate RMSE and MAE
    train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
    test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))
    train_mae = mean_absolute_error(Y_train, train_predict)
    test_mae = mean_absolute_error(Y_test, test_predict)

    print(f'Train RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')
    print(f'Train MAE: {train_mae}')
    print(f'Test MAE: {test_mae}')

    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(Y_train)), scaler.inverse_transform(Y_train.reshape(-1, 1)), label='Actual Train')
    plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), scaler.inverse_transform(Y_test.reshape(-1, 1)), label='Actual Test')
    plt.plot(range(len(Y_train)), train_predict, label='Predicted Train')
    plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), test_predict, label='Predicted Test')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Residual analysis
    train_residuals = Y_train - train_predict.flatten()
    test_residuals = Y_test - test_predict.flatten()

    plt.figure(figsize=(14, 7))
    plt.plot(range(len(train_residuals)), train_residuals, label='Train Residuals')
    plt.plot(range(len(train_residuals), len(train_residuals) + len(test_residuals)), test_residuals, label='Test Residuals')
    plt.title('Residuals Analysis')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model('stock_price.csv')