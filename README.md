# AI-Based Stock Price Prediction with LSTM

## Overview
This project leverages a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. It encompasses data preprocessing, feature engineering, model training, and evaluation to ensure accurate predictions.

## Project Structure
```
├── pre.py          # Data preprocessing functions
├── train.py        # Feature engineering and model training
├── evaluate.py     # Model evaluation
├── main.py         # Executes the full prediction pipeline sequentially
├── README.md       # Project documentation
└── requirements.txt # List of dependencies
```

## Requirements
Ensure you have the following dependencies installed before running the project:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Usage
### 1. Data Preprocessing
The `pre.py` script contains the `preprocess_data` function, which prepares and cleans the stock price data for model training.

Run the preprocessing script:
```bash
python pre.py
```

### 2. Feature Engineering
The `train.py` script includes the `add_features` function, which introduces additional features such as moving averages and volatility to enhance the model's predictive capability.

### 3. Model Training
The `train.py` script also contains the `train_model` function, responsible for training the LSTM model using the processed dataset.

Run the training script:
```bash
python train.py
```

### 4. Model Evaluation
The `evaluate.py` script includes the `evaluate_model` function, which assesses the trained model's performance using appropriate metrics.

Run the evaluation script:
```bash
python evaluate.py
```

### 5. Running the Full Pipeline
To execute the entire workflow sequentially, run the `main.py` script:
```bash
python main.py
```

## Results & Visualization
- The trained LSTM model provides insights into stock price trends.
- Data visualizations are generated using Matplotlib and Seaborn to help analyze stock market movements and model performance.

## Technical Highlights
- Built using TensorFlow and Keras for deep learning.
- Data preprocessing and feature engineering powered by pandas and scikit-learn.
- Advanced data visualization with Matplotlib and Seaborn.

## Future Improvements
- Enhancing the model with additional financial indicators.
- Implementing hyperparameter tuning for better performance.
- Deploying the model as a web-based application for real-time predictions.


