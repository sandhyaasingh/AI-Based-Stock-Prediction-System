# Stock Price Prediction with LSTM

This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The project includes data preprocessing, feature engineering, model training, and evaluation.
Note: Changed the column names from Japanese to English (included here).

## Project Structure

- `pre.py`: Contains functions for data preprocessing.
- `train.py`: Contains functions for feature engineering and model training.
- `evaluate.py`: Contains functions for model evaluation.
- `main.py`: For executing the prediction system sequentially.
- `README.md`: Project documentation.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Usage

### Data Preprocessing

The `pre.py` file contains the `preprocess_data` function, which preprocesses the stock price data.

### Feature Engineering

The `train.py` file contains the `add_features` function, which adds moving averages and volatility as features.

### Model Training

The `train.py` file contains the `train_model` function, which trains the LSTM model.

### Model Evaluation

The `evaluate.py` file contains the `evaluate_model` function, which evaluates the trained model.

## Example

Here is an example of how to run the entire pipeline:
```bash
python pre.py
python train.py
python evaluate.py
```
or you may run the main.py to run it all sequentially.
```bash
python main.py
```

## Acknowledgements

- The LSTM model is built using TensorFlow and Keras.
- Data preprocessing and feature engineering are done using pandas and scikit-learn.
- Visualization is done using matplotlib and seaborn.