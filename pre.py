import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    # Load the CSV data
    df = pd.read_csv(file_path)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Rename columns to match Prophet's requirements
    df.rename(columns={'Date': 'ds', 'closing-price': 'y'}, inplace=True)

    # Convert 'ds' to datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Ensure 'y' is treated as a string, then remove commas and 'M'
    df['y'] = df['y'].astype(str).str.replace('M', '').str.replace(',', '')

    # Convert 'y' to numeric
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop rows with missing values
    df.dropna(subset=['ds', 'y'], inplace=True)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['y']])

    # Create a DataFrame with the scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=['y'], index=df['ds'])

    return scaled_df, scaler

if __name__ == "__main__":
    scaled_df, scaler = preprocess_data('stock_price.csv')
    print(scaled_df.head())