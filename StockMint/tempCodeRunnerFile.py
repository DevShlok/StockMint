import os
import io
import base64
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from keras.models import load_model
from flask import Flask, render_template, request, send_file
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = Flask(__name__)

# Load models once at startup
stock_model = load_model('Stock_Price_Prediction.h5')
crypto_model = load_model('Crypto_Model.keras')

# ------------- Helper Functions ------------------
def plot_to_base64_png(fig):
    """Convert matplotlib figure to base64 PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def create_ema_plot(df, spans, colors, labels, title):
    """Create EMA plot with optimized layout"""
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=100, constrained_layout=True)
    ax.plot(df.index, df['Close'], color='blue', linewidth=1, label='Closing Price')
    
    for span, color, label in zip(spans, colors, labels):
        ema = df['Close'].ewm(span=span, adjust=False).mean()
        ax.plot(df.index, ema, color=color, linewidth=1, label=label)
    
    ax.set_title(title, pad=15)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Price", labelpad=10)
    ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=25)
    return fig

def safe_int(value, default=10):
    """Safely convert to integer with fallback"""
    try: return int(value)
    except: return default

# ---------------- Routes ------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'asset_type': None, 'symbol': None, 'days': None, 'error': None,
        'data_desc': None, 'hist_stock_plot': None, 'ema_20_50_plot': None,
        'ema_100_200_plot': None, 'prediction_plot': None, 'original_plot': None,
        'predicted_plot': None, 'future_plot': None, 'future_predictions': None,
        'dataset_link': None
    }

    if request.method == 'POST':
        # Form handling
        asset_type = request.form.get('asset_type')
        symbol = request.form.get('symbol', '').upper().strip()
        days = safe_int(request.form.get('days', 10), 10)
        
        context.update(asset_type=asset_type, symbol=symbol, days=days)
        
        if not symbol:
            context['error'] = "Please enter a valid ticker symbol."
            return render_template('index.html', **context)

        # Data fetching
        try:
            df = yf.download(symbol, start=dt.datetime(2010,1,1), end=dt.datetime.now())
            if df.empty:
                context['error'] = "No data found for symbol."
                return render_template('index.html', **context)
        except Exception as e:
            context['error'] = f"Data fetch error: {str(e)}"
            return render_template('index.html', **context)

        # Save CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_file_path = f"static/{symbol}_dataset.csv"
        with open(csv_file_path, 'w') as f:
            f.write(csv_buffer.getvalue())
        context['dataset_link'] = csv_file_path
        context['data_desc'] = df.describe().to_html(classes='table table-hover')

        if asset_type == 'stock':
            # Stock plots
            context['hist_stock_plot'] = plot_to_base64_png(
                create_ema_plot(df, [1], ['blue'], ['Close'], "Historical Closing Prices")
            )
            context['ema_20_50_plot'] = plot_to_base64_png(
                create_ema_plot(df, [20,50], ['green','red'], ['EMA 20','EMA 50'], "20 & 50 Day EMAs")
            )
            context['ema_100_200_plot'] = plot_to_base64_png(
                create_ema_plot(df, [100,200], ['purple','orange'], ['EMA 100','EMA 200'], "100 & 200 Day EMAs")
            )

            # Stock predictions
            train_data = df['Close'][:int(len(df)*0.7)]
            scaler = MinMaxScaler(feature_range=(0,1)).fit(train_data.values.reshape(-1,1))
            
            test_data = pd.concat([df['Close'].tail(100), df['Close'][int(len(df)*0.7):]])
            scaled_test = scaler.transform(test_data.values.reshape(-1,1))
            
            X_test = np.array([scaled_test[i-100:i] for i in range(100, len(scaled_test))])
            y_pred = stock_model.predict(X_test, verbose=0) * (1/scaler.scale_[0])
            y_true = scaled_test[100:] * (1/scaler.scale_[0])

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_true.flatten(), name='True', line=dict(color='green')))
            fig.add_trace(go.Scatter(y=y_pred.flatten(), name='Predicted', line=dict(color='red')))
            fig.update_layout(title='Price Prediction', height=350, template='plotly_white')
            context['prediction_plot'] = fig.to_html(full_html=False)

        elif asset_type == 'crypto':
            # Crypto plots
            context['original_plot'] = plot_to_base64_png(
                create_ema_plot(df, [1], ['blue'], ['Close'], "Historical Prices")
            )

            # Crypto predictions
            test_data = df['Close'][int(len(df)*0.9):].values.reshape(-1,1)
            scaler = MinMaxScaler(feature_range=(0,1)).fit(test_data)
            scaled_data = scaler.transform(test_data)
            
            X, y = [], []
            for i in range(100, len(scaled_data)):
                X.append(scaled_data[i-100:i])
                y.append(scaled_data[i])
                
            y_pred = scaler.inverse_transform(crypto_model.predict(np.array(X), verbose=0))
            y_true = scaler.inverse_transform(y)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_true.flatten(), name='True', line=dict(color='green')))
            fig.add_trace(go.Scatter(y=y_pred.flatten(), name='Predicted', line=dict(color='red')))
            fig.update_layout(title='Price Prediction', height=350, template='plotly_white')
            context['predicted_plot'] = fig.to_html(full_html=False)

            # Future prediction
            last_100 = scaler.transform(df['Close'].tail(100).values.reshape(-1,1))
            future = []
            current_input = last_100.reshape(1, -1, 1)
            
            for _ in range(days):
                next_pred = crypto_model.predict(current_input, verbose=0)
                future.append(scaler.inverse_transform(next_pred)[0,0])
                current_input = np.append(current_input[:,1:,:], next_pred.reshape(1,1,-1), axis=1)
                
            context['future_predictions'] = future
            fig = go.Figure(go.Scatter(x=list(range(1, days+1)), y=future, mode='lines+markers'))
            fig.update_layout(title=f'{days} Day Forecast', height=350, template='plotly_white')
            context['future_plot'] = fig.to_html(full_html=False)

    return render_template('index.html', **context)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
