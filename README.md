# StockMint


StockMint is a Flask web application for forecasting and visualizing stock and cryptocurrency price trends. It uses deep learning models, yfinance for data, and generates interactive and static graphs (Plotly and Matplotlib) for both historical and predicted price analysis.

Features
Predict Stock and Crypto price trends

Interactive web interface (Flask + Bootstrap)

Supports Stocks and Crypto assets

Stock & Crypto symbol lookup (e.g., AAPL, BTC-USD)

Visualization:

Matplotlib charts for Historical prices and EMAs (20, 50, 100, 200 days)

Plotly for interactive prediction comparisons

Summary statistics and dataset download

Future price forecasting for cryptocurrencies


Requirements
Python 3.8+

See requirements.txt (sample below)

Example requirements.txt:
text
Flask
pandas
numpy
yfinance
keras
scikit-learn
matplotlib
plotly
Getting Started
1. Clone Repository
bash
git clone https://github.com/yourusername/asset-trend-predictor.git
cd asset-trend-predictor
2. Install Dependencies
It's recommended to use a virtual environment.

bash
pip install -r requirements.txt
3. Place Model Files
Place your model files in the root directory:

Stock_Price_Prediction.h5 (for stocks)

Crypto_Model.keras (for crypto)

(You can use your own pre-trained models.)

4. Run the App
bash
python app.py
The app will be available at http://127.0.0.1:5000/.

Usage
Open the web app in your browser.

Select Asset Type: Stock or Cryptocurrency.

Enter the asset symbol (e.g., AAPL for Apple, BTC-USD for Bitcoin).

(For Crypto) Enter the number of days to predict.

Click Analyze.

View historical, EMA, and prediction charts, as well as descriptive statistics and dataset download options.

Project Structure
text
.
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── ... (for generated CSV downloads)
├── requirements.txt
├── Stock_Price_Prediction.h5
├── Crypto_Model.keras
└── README.md
Customization
To use different models, swap the Stock_Price_Prediction.h5 or Crypto_Model.keras files.

Charts are sized to 900x350px for both Plotly and Matplotlib for consistency.

You can adjust theme/colors in index.html or Matplotlib code sections.

License
MIT License (or specify your license)

Credits
Developed by Shlok Shukla

Powered by Flask, Keras, yfinance, Matplotlib, and Plotly

Happy Predicting! 📈
