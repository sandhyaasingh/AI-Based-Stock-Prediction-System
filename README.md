# AI-Based Stock Price Prediction System

This project is a Streamlit-based web application that predicts the future stock prices of top tech companies using LSTM (Long Short-Term Memory) neural networks.

## 🔍 Project Overview

* **Frameworks/Libraries:** Python, Streamlit, Keras, scikit-learn, Matplotlib, yfinance
* **Model:** LSTM (Recurrent Neural Network)
* **Features:**

  * Real-time data fetching using Yahoo Finance (yfinance)
  * Time series preprocessing using MinMaxScaler
  * Sequence generation (past 60 days to predict the next)
  * Model trained dynamically each time you run
  * Forecasts the next 10 days
  * Visualizations for:

    * Actual vs. Predicted historical data
    * Future stock forecast

## 📈 Companies Supported

* Apple (AAPL)
* Google (GOOGL)
* Microsoft (MSFT)
* Amazon (AMZN)
* Tesla (TSLA)
* Meta (META)

## 📁 Files in This Repo

| File               | Description                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| `main.py`          | Base Python file to train LSTM and visualize for one hardcoded stock (for debugging or reference). |
| `app.py`           | Main web app file built using Streamlit that allows company selection and shows predictions.       |
| `requirements.txt` | All dependencies to run the project.                                                               |
| `.gitignore`       | Prevents unnecessary files like `venv/` from being pushed to GitHub.                               |

---

## ▶️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/sandhyaasingh/AI-Based-Stock-Prediction-System.git
cd AI-Based-Stock-Prediction-System
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📲 Access on Mobile

Once deployed (e.g., on Streamlit Cloud), you can:

* Open it from any browser (phone or laptop)
* Share the link with friends
* Predictions will always be **up-to-date** with the latest stock data

---

## 🤖 How It Works (Simplified)

1. **Collect data:** Download past 2.5 years of stock price data (close price).
2. **Scale the data:** Normalize using `MinMaxScaler`.
3. **Prepare sequences:** Take the past 60 days to predict the 61st.
4. **Train LSTM model:** Two LSTM layers followed by a dense output.
5. **Predict and inverse-scale:** Show actual vs predicted.
6. **Forecast 10 future days:** Using the latest sequence.


---


## ⭐ If You Like It

Give this repo a ⭐ on GitHub to support!

---
