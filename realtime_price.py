import yfinance as yf


def get_live_price(stock_symbol):
    try:
        data = yf.download(stock_symbol, period="1d", interval="1m", auto_adjust=True, progress=False)
        data.dropna(inplace=True)
        if data.empty:
            return None
        return round(float(data['Close'].iloc[-1]), 2)
    except Exception:
        return None
