import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    """
    Perform Exploratory Data Analysis (EDA) on the given CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    None
    """
    # Load the CSV data
    df = pd.read_csv(file_path)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Print column names to verify
    print(df.columns)

    # Display basic statistics
    print(df.describe())

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Convert non-numeric values to numeric
    for column in df.columns:
        df[column] = df[column].astype(str).str.replace('M', '').str.replace(',', '')
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Plot closing price over time
    plt.figure(figsize=(14, 7))
    plt.plot(df['closing-price'])
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()

    # Plot correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    perform_eda('stock_price.csv')