from eda import perform_eda
from pre import preprocess_data
from train import train_model
from evaluate import evaluate_model

def main():
    file_path = 'stock_price.csv'
    
    # Step 1: Data Understanding and EDA
    perform_eda(file_path)
    
    # Step 2: Data Preprocessing and Feature Engineering
    scaled_df, scaler = preprocess_data(file_path)
    
    # Step 3: Model Selection and Training
    model, X_train, X_test, Y_train, Y_test, scaler = train_model(file_path)
    
    # Step 4: Model Evaluation and Results Analysis
    evaluate_model(file_path)

if __name__ == "__main__":
    main()