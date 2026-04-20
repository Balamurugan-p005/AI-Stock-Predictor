import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)
import tensorflow as tf
from tensorflow.keras.models   import Model
from tensorflow.keras.layers   import (
    Input, Conv1D, MaxPooling1D, Dropout,
    LSTM, Bidirectional, Dense,
    BatchNormalization, Layer
)
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend  as K


# ═══════════════════════════════════════════════
# ATTENTION LAYER
# ═══════════════════════════════════════════════
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)   # (batch, time, 1)
        a = K.softmax(e, axis=1)                  # (batch, time, 1)
        return K.sum(x * a, axis=1)               # (batch, units)

    def compute_output_shape(self, s):
        return (s[0], s[-1])


# ═══════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════
def add_indicators(df):
    # Safely extract each column as plain 1D float Series
    def get_col(name):
        col = df[name].copy()
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col.astype(float)

    c = get_col('Close')
    h = get_col('High')  if 'High'   in df.columns else c
    l = get_col('Low')   if 'Low'    in df.columns else c
    v = get_col('Volume')if 'Volume' in df.columns else pd.Series(1, index=c.index)

    df['SMA_10'] = c.rolling(10).mean()
    df['SMA_20'] = c.rolling(20).mean()
    df['EMA_12'] = c.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = c.ewm(span=26, adjust=False).mean()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    df['MACD_S'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta        = c.diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']    = 100 - 100 / (1 + gain / (loss + 1e-9))

    sma20        = c.rolling(20).mean()
    std20        = c.rolling(20).std()
    bb_u         = sma20 + 2 * std20
    bb_l         = sma20 - 2 * std20
    df['BB_W']   = bb_u - bb_l
    df['BB_P']   = (c - bb_l) / (df['BB_W'] + 1e-9)

    tr           = pd.concat([
                       (h - l),
                       (h - c.shift(1)).abs(),
                       (l - c.shift(1)).abs()
                   ], axis=1).max(axis=1)
    df['ATR']    = tr.rolling(14).mean()

    vol_ma       = v.rolling(20).mean()
    df['VRATIO'] = v / (vol_ma + 1e-9)
    df['PCT']    = c.pct_change()
    df['ROC']    = c.pct_change(10)

    return df


# ═══════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════
def get_stock_data(symbol, period="5y"):
    df = yf.download(symbol, period=period,
                     auto_adjust=True, progress=False)

    # ── Flatten MultiIndex columns ────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"[Data] Raw download: {len(df)} rows, columns: {list(df.columns)}")

    # ── Keep only needed columns ──────────────
    needed = ['Open','High','Low','Close','Volume']
    df = df[[c for c in needed if c in df.columns]].copy()

    # ── Convert to float safely ───────────────
    for col in df.columns:
        try:
            vals = df[col]
            if hasattr(vals, 'squeeze'):
                vals = vals.squeeze()
            df[col] = pd.to_numeric(vals, errors='coerce')
        except Exception:
            pass

    # ── Only remove rows where Close is NaN or zero ──
    df = df[df['Close'].notna()]
    df = df[df['Close'] > 0]

    # Add Volume if missing
    if 'Volume' not in df.columns:
        df['Volume'] = 1000000

    # Fill Volume NaN with median
    df['Volume'] = df['Volume'].fillna(df['Volume'].median())
    df['Volume'] = df['Volume'].replace(0, df['Volume'].median())

    # Fill other NaN with forward fill
    df = df.ffill().bfill()

    print(f"[Data] After cleaning: {len(df)} rows")

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} rows for {symbol}. "
            "Check internet connection and symbol spelling."
        )

    df = add_indicators(df)

    # Only drop rows where Close indicator is NaN
    df = df[df['Close'].notna()]
    df = df.ffill().bfill()
    df.dropna(inplace=True)

    print(f"[Data] After indicators: {len(df)} rows")
    return df


# ═══════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════
def build_model(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x   = BatchNormalization()(x)
    x   = MaxPooling1D(2)(x)
    x   = Dropout(0.2)(x)
    x   = Bidirectional(LSTM(128, return_sequences=True))(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.3)(x)
    x   = Bidirectional(LSTM(64, return_sequences=True))(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.2)(x)
    x   = AttentionLayer()(x)
    x   = Dense(64, activation='relu')(x)
    x   = Dropout(0.1)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss='huber', metrics=['mae'])
    return m


# ═══════════════════════════════════════════════
# INVERSE TRANSFORM HELPER  ← THE KEY FIX
# ═══════════════════════════════════════════════
def inverse_close(scaler, values_1d, n_feat, close_col):
    """
    Correctly inverse-transform scaled Close prices
    using the SAME scaler that was used for training.

    We fill a dummy array with the scaled values in the
    Close column and zeros elsewhere, then inverse_transform,
    then extract the Close column.

    This guarantees the same min/max range is used in both
    directions — eliminating the scaler mismatch bug.
    """
    dummy = np.zeros((len(values_1d), n_feat))
    dummy[:, close_col] = values_1d.flatten()
    return scaler.inverse_transform(dummy)[:, close_col].reshape(-1, 1)


# ═══════════════════════════════════════════════
# MAIN: TRAIN + PREDICT
# ═══════════════════════════════════════════════
def train_and_predict_lstm(symbol, sequence_length=40):

    # 1. Data
    df = get_stock_data(symbol, period="5y")
    if len(df) < sequence_length + 60:
        raise ValueError(
            f"Not enough data ({len(df)} rows) for {symbol}. "
            f"Need at least {sequence_length + 60} rows. "
            "Try: 1) Use 5y period  2) Reduce sequence length to 30  "
            "3) Check symbol is correct (e.g. TCS.NS not TCS)"
        )

    # 2. Feature list
    FEAT = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_S', 'RSI',
        'BB_W', 'BB_P', 'ATR', 'VRATIO', 'PCT', 'ROC'
    ]
    FEAT      = [f for f in FEAT if f in df.columns]
    close_col = FEAT.index('Close')   # always 0 with above list
    n_feat    = len(FEAT)

    raw = df[FEAT].values.astype(float)

    # 3. ── SINGLE SCALER ──────────────────────────
    #    ONE scaler for ALL features.
    #    y values come from scaled[:, close_col].
    #    Inverse transform uses the SAME scaler via
    #    the dummy-array trick in inverse_close().
    #    This eliminates the scale-mismatch bug.
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(raw)   # shape (N, n_feat)

    # 4. Sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length:i, :])
        y.append(scaled[i, close_col])          # ← scaled by SAME scaler
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 5. Chronological split  75 / 10 / 15
    n     = len(X)
    t_end = int(0.75 * n)
    v_end = int(0.85 * n)

    Xtr, ytr = X[:t_end],       y[:t_end]
    Xvl, yvl = X[t_end:v_end],  y[t_end:v_end]
    Xte, yte = X[v_end:],       y[v_end:]

    if len(Xte) < 20:
        raise ValueError("Test set too small. Increase period to 5y.")

    # 6. Train
    model = build_model(sequence_length, n_feat)
    cbs   = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0),
    ]
    model.fit(
        Xtr, ytr,
        epochs=150, batch_size=32,
        validation_data=(Xvl, yvl),
        callbacks=cbs,
        shuffle=False, verbose=0,
    )

    # 7. Evaluate — inverse transform with SAME scaler
    yp_sc  = model.predict(Xte, verbose=0).flatten()   # scaled
    ya_sc  = yte                                        # scaled

    yp_real = inverse_close(scaler, yp_sc, n_feat, close_col)
    ya_real = inverse_close(scaler, ya_sc, n_feat, close_col)

    mae   = mean_absolute_error(ya_real, yp_real)
    rmse  = np.sqrt(mean_squared_error(ya_real, yp_real))
    r2    = r2_score(ya_real, yp_real)
    mape  = float(np.mean(
                np.abs((ya_real - yp_real) / (np.abs(ya_real) + 1e-9))
            )) * 100
    ad    = np.sign(np.diff(ya_real.flatten()))
    pd_   = np.sign(np.diff(yp_real.flatten()))
    dacc  = float(np.mean(ad == pd_)) * 100

    metrics = {
        "MAE"                    : round(float(mae),  4),
        "RMSE"                   : round(float(rmse), 4),
        "R2"                     : round(float(r2),   4),
        "Direction Accuracy (%)" : round(dacc,        2),
        "MAPE (%)"               : round(mape,        2),
    }

    # 8. Next-day prediction — inverse transform with SAME scaler
    last_seq    = scaled[-sequence_length:].reshape(1, sequence_length, n_feat)
    next_sc     = model.predict(last_seq, verbose=0).flatten()
    next_price  = inverse_close(scaler, next_sc, n_feat, close_col)[0][0]

    return (
        round(float(next_price), 2),
        df,
        metrics,
        ya_real.flatten().tolist(),
        yp_real.flatten().tolist(),
    )