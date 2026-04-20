"""
confidence_score.py
────────────────────────────────────────────
Calculates an overall Confidence / Trust Score (0–100)
by combining 5 independent signals:

  1. LSTM Model Accuracy     (30 pts) — how well the model performs
  2. Sentiment Strength      (20 pts) — how strong/clear the news signal is
  3. Price Prediction Signal (20 pts) — how significant the predicted growth is
  4. Volatility Risk         (15 pts) — penalize highly volatile stocks
  5. Data Quality            (15 pts) — enough data, recent, no gaps

Final score → label:
  80–100 : VERY HIGH  🟢
  65–79  : HIGH       🟢
  50–64  : MODERATE   🟡
  35–49  : LOW        🟠
  0–34   : VERY LOW   🔴
"""

import numpy as np


# ─────────────────────────────────────────────
# Individual Signal Scorers
# ─────────────────────────────────────────────

def score_model_accuracy(metrics: dict) -> tuple[float, dict]:
    """
    Max 30 points.
    Uses R², Direction Accuracy, and MAPE together.
    """
    r2       = metrics.get("R2", 0)
    dir_acc  = metrics.get("Direction Accuracy (%)", 50)
    mape     = metrics.get("MAPE (%)", 10)

    # R² score (0–12 pts)
    if r2 >= 0.95:    r2_pts = 12
    elif r2 >= 0.90:  r2_pts = 10
    elif r2 >= 0.85:  r2_pts = 8
    elif r2 >= 0.75:  r2_pts = 5
    else:             r2_pts = 2

    # Direction accuracy (0–12 pts)
    if dir_acc >= 68:    da_pts = 12
    elif dir_acc >= 63:  da_pts = 10
    elif dir_acc >= 58:  da_pts = 7
    elif dir_acc >= 53:  da_pts = 4
    else:                da_pts = 1

    # MAPE (0–6 pts) — lower MAPE = more points
    if mape <= 1.5:   mape_pts = 6
    elif mape <= 2.5: mape_pts = 5
    elif mape <= 4.0: mape_pts = 3
    elif mape <= 6.0: mape_pts = 1
    else:             mape_pts = 0

    total = r2_pts + da_pts + mape_pts

    breakdown = {
        "R² Score"            : {"score": r2_pts,   "max": 12, "value": f"{r2:.4f}"},
        "Direction Accuracy"  : {"score": da_pts,   "max": 12, "value": f"{dir_acc:.1f}%"},
        "MAPE"                : {"score": mape_pts, "max": 6,  "value": f"{mape:.2f}%"},
    }
    return total, breakdown


def score_sentiment(sentiment_label: str, sentiment_score: float,
                    sentiment_detail: list) -> tuple[float, dict]:
    """
    Max 20 points.
    Rewards strong, clear sentiment signals.
    """
    # Base points from label
    if sentiment_label == "Positive":
        label_pts = 14
    elif sentiment_label == "Neutral":
        label_pts = 8
    else:  # Negative
        label_pts = 3

    # Confidence bonus (avg confidence of FinBERT predictions)
    avg_conf = 0
    if sentiment_detail:
        avg_conf = np.mean([d.get("confidence", 0) for d in sentiment_detail])

    if avg_conf >= 0.85:   conf_pts = 6
    elif avg_conf >= 0.70: conf_pts = 4
    elif avg_conf >= 0.55: conf_pts = 2
    else:                  conf_pts = 0

    total = label_pts + conf_pts

    breakdown = {
        "Sentiment Label"     : {"score": label_pts, "max": 14, "value": sentiment_label},
        "FinBERT Confidence"  : {"score": conf_pts,  "max": 6,  "value": f"{avg_conf:.2f}"},
    }
    return total, breakdown


def score_price_signal(predicted_price: float, live_price: float) -> tuple[float, dict]:
    """
    Max 20 points.
    Rewards clear, significant growth predictions.
    Penalizes contradictory signals.
    """
    if live_price == 0:
        return 10, {"Growth Signal": {"score": 10, "max": 20, "value": "N/A"}}

    growth = ((predicted_price - live_price) / live_price) * 100
    abs_g  = abs(growth)

    # Strong clear signal = high confidence
    if abs_g >= 5:    signal_pts = 20
    elif abs_g >= 3:  signal_pts = 16
    elif abs_g >= 1:  signal_pts = 11
    elif abs_g >= 0:  signal_pts = 7
    else:             signal_pts = 4   # Negative growth signal

    breakdown = {
        "Predicted Growth"    : {"score": signal_pts, "max": 20, "value": f"{growth:+.2f}%"},
    }
    return signal_pts, breakdown


def score_volatility(hist_data) -> tuple[float, dict]:
    """
    Max 15 points.
    Low volatility stocks = more predictable = higher confidence.
    Uses 30-day rolling standard deviation of daily returns.
    """
    try:
        close    = hist_data['Close'].values.flatten()
        returns  = np.diff(close) / close[:-1]
        vol_30d  = np.std(returns[-30:]) * 100   # % daily volatility

        if vol_30d <= 1.0:   vol_pts = 15
        elif vol_30d <= 1.5: vol_pts = 12
        elif vol_30d <= 2.5: vol_pts = 9
        elif vol_30d <= 4.0: vol_pts = 5
        else:                vol_pts = 2

        breakdown = {
            "30-Day Volatility"   : {"score": vol_pts, "max": 15, "value": f"{vol_30d:.2f}% daily"},
        }
        return vol_pts, breakdown
    except Exception:
        return 7, {"30-Day Volatility": {"score": 7, "max": 15, "value": "N/A"}}


def score_data_quality(hist_data, metrics: dict) -> tuple[float, dict]:
    """
    Max 15 points.
    More data rows + low RMSE relative to price = better quality.
    """
    try:
        n_rows   = len(hist_data)
        close    = hist_data['Close'].values.flatten()
        avg_price = np.mean(close)
        rmse     = metrics.get("RMSE", 999)
        rmse_pct = (rmse / avg_price) * 100   # RMSE as % of avg price

        # Data volume (0–8 pts)
        if n_rows >= 700:    row_pts = 8
        elif n_rows >= 500:  row_pts = 6
        elif n_rows >= 300:  row_pts = 4
        elif n_rows >= 150:  row_pts = 2
        else:                row_pts = 0

        # RMSE % (0–7 pts)
        if rmse_pct <= 1.0:   rmse_pts = 7
        elif rmse_pct <= 2.0: rmse_pts = 5
        elif rmse_pct <= 4.0: rmse_pts = 3
        elif rmse_pct <= 6.0: rmse_pts = 1
        else:                  rmse_pts = 0

        total = row_pts + rmse_pts
        breakdown = {
            "Data Volume"         : {"score": row_pts,  "max": 8, "value": f"{n_rows} days"},
            "RMSE % of Price"     : {"score": rmse_pts, "max": 7, "value": f"{rmse_pct:.2f}%"},
        }
        return total, breakdown
    except Exception:
        return 7, {"Data Quality": {"score": 7, "max": 15, "value": "N/A"}}


# ─────────────────────────────────────────────
# MASTER CONFIDENCE SCORE
# ─────────────────────────────────────────────

def calculate_confidence_score(
    metrics: dict,
    sentiment_label: str,
    sentiment_score: float,
    sentiment_detail: list,
    predicted_price: float,
    live_price: float,
    hist_data
) -> dict:
    """
    Returns full confidence report:
    {
        score       : 0–100 int,
        label       : "VERY HIGH" | "HIGH" | "MODERATE" | "LOW" | "VERY LOW",
        color       : hex color,
        emoji       : emoji,
        breakdown   : { signal_name: {score, max, value} },
        advice      : human-readable string,
        gauge_color : for plotly gauge
    }
    """

    # Score each signal
    s1, b1 = score_model_accuracy(metrics)
    s2, b2 = score_sentiment(sentiment_label, sentiment_score, sentiment_detail)
    s3, b3 = score_price_signal(predicted_price, live_price)
    s4, b4 = score_volatility(hist_data)
    s5, b5 = score_data_quality(hist_data, metrics)

    total_score = s1 + s2 + s3 + s4 + s5   # max 100

    # Combine all breakdowns
    full_breakdown = {}
    full_breakdown.update(b1)
    full_breakdown.update(b2)
    full_breakdown.update(b3)
    full_breakdown.update(b4)
    full_breakdown.update(b5)

    # Section subtotals
    section_scores = {
        "🧠 Model Accuracy"    : {"earned": s1, "max": 30},
        "📰 Sentiment Signal"  : {"earned": s2, "max": 20},
        "📈 Price Signal"      : {"earned": s3, "max": 20},
        "📉 Volatility Risk"   : {"earned": s4, "max": 15},
        "📊 Data Quality"      : {"earned": s5, "max": 15},
    }

    # Label + color
    if total_score >= 80:
        label       = "VERY HIGH"
        color       = "#00ff88"
        emoji       = "🟢"
        gauge_color = "green"
        advice      = "Strong buy/sell signal. All indicators align. High trust in this prediction."
    elif total_score >= 65:
        label       = "HIGH"
        color       = "#66ff99"
        emoji       = "🟢"
        gauge_color = "green"
        advice      = "Good signal strength. Most indicators agree. Reasonable to act on this prediction."
    elif total_score >= 50:
        label       = "MODERATE"
        color       = "#ffff00"
        emoji       = "🟡"
        gauge_color = "yellow"
        advice      = "Mixed signals. Some indicators conflict. Use alongside other research before deciding."
    elif total_score >= 35:
        label       = "LOW"
        color       = "#ff8800"
        emoji       = "🟠"
        gauge_color = "orange"
        advice      = "Weak signal. High uncertainty. Consider waiting for clearer market conditions."
    else:
        label       = "VERY LOW"
        color       = "#ff4444"
        emoji       = "🔴"
        gauge_color = "red"
        advice      = "Very unreliable prediction. Do not make trading decisions based on this alone."

    return {
        "score"          : total_score,
        "label"          : label,
        "color"          : color,
        "emoji"          : emoji,
        "gauge_color"    : gauge_color,
        "advice"         : advice,
        "breakdown"      : full_breakdown,
        "section_scores" : section_scores,
    }