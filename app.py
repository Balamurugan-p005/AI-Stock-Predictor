import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from auth import init_db, login_user, register_user, save_search, get_search_history
from prediction import train_and_predict_lstm, get_stock_data
from realtime_price import get_live_price
from sentiment import analyze_sentiment
from news_fetcher import get_stock_news, get_insider_sentiment
from confidence_score import calculate_confidence_score

init_db()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { background-color: #0d1117; color: #e6edf3; }
    .main-header {
        font-size: 2.6rem; font-weight: 900;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .auth-container {
        max-width: 420px; margin: auto; background: #161b22;
        border: 1px solid #30363d; border-radius: 16px; padding: 2.5rem 2rem;
    }
    .auth-title {
        font-size: 1.6rem; font-weight: 800;
        text-align: center; margin-bottom: 1.5rem; color: #00d2ff;
    }
    .user-badge {
        background: linear-gradient(135deg,#1f2937,#111827);
        border:1px solid #374151; border-radius:10px;
        padding:0.6rem 1rem; font-size:0.9rem; color:#9ca3af;
    }
    .recommendation-box {
        border-radius:14px; padding:1.5rem; text-align:center;
        font-size:2rem; font-weight:900; letter-spacing:2px;
    }
    .reco-STRONG-BUY { background:#004d26; color:#00ff88; border:2px solid #00ff88; }
    .reco-BUY        { background:#003d1a; color:#66ff99; border:2px solid #66ff99; }
    .reco-HOLD       { background:#3d3d00; color:#ffff00; border:2px solid #ffff00; }
    .reco-SELL       { background:#4d0000; color:#ff4444; border:2px solid #ff4444; }
    .reco-AVOID      { background:#4d1a00; color:#ff8800; border:2px solid #ff8800; }
    .reco-NOT-BUY    { background:#4d0000; color:#ff4444; border:2px solid #ff4444; }
    .accuracy-bar    { border-radius:8px; background:#0d1117; padding:0.8rem 1rem; margin-bottom:0.5rem; }
    .history-item    { background:#161b22; border:1px solid #30363d; border-radius:8px;
                       padding:0.4rem 0.8rem; margin-bottom:0.3rem; font-size:0.85rem; }
    .confidence-box  {
        border-radius:16px; padding:1.8rem; text-align:center;
        margin: 1rem 0;
    }
    .signal-row {
        display:flex; justify-content:space-between; align-items:center;
        background:#161b22; border:1px solid #30363d; border-radius:8px;
        padding:0.5rem 1rem; margin-bottom:0.4rem; font-size:0.9rem;
    }
    .signal-bar-fill {
        height:8px; border-radius:4px; background: linear-gradient(90deg,#00d2ff,#3a7bd5);
    }
    div[data-testid="stForm"] { border:none !important; padding:0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "logged_in"  not in st.session_state: st.session_state.logged_in  = False
if "user"       not in st.session_state: st.session_state.user       = None
if "auth_page"  not in st.session_state: st.session_state.auth_page  = "login"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_recommendation(predicted, live, sentiment, insider="Neutral", investor_type="New Investor"):
    if live is None or live == 0: return "HOLD", 0.0
    growth = ((predicted - live) / live) * 100

    if investor_type == "New Investor":
        # New investor — should they enter the market?
        if sentiment == "Negative" and insider != "Bullish":
            reco = "NOT BUY"
        elif growth >= 5 and insider != "Bearish":
            reco = "BUY"
        elif growth >= 1:
            reco = "HOLD"
        elif growth >= -2:
            reco = "HOLD"
        else:
            reco = "NOT BUY"
        # Insider boost for new investor
        if insider == "Bullish" and reco == "HOLD":
            reco = "BUY"
        if insider == "Bearish" and reco == "BUY":
            reco = "HOLD"

    else:
        # Existing investor — should they keep, add or exit?
        if sentiment == "Negative" and insider != "Bullish":
            reco = "SELL"
        elif growth >= 5 and insider != "Bearish":
            reco = "BUY"       # add more shares
        elif growth >= 0:
            reco = "HOLD"      # keep current position
        else:
            reco = "SELL"      # exit to avoid loss
        # Insider signals
        if insider == "Bullish" and reco == "HOLD":
            reco = "BUY"
        if insider == "Bearish" and reco in ["BUY"]:
            reco = "HOLD"

    return reco, round(growth, 2)


def get_currency(stock_symbol):
    if ".NS" in stock_symbol or ".BO" in stock_symbol:
        return "₹"
    elif ".L"  in stock_symbol: return "£"
    elif ".PA" in stock_symbol or ".DE" in stock_symbol: return "€"
    else: return "$"


def render_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": "Trust Score", "font": {"color": "#fff", "size": 18}},
        number= {"font": {"color": color, "size": 48}, "suffix": "/100"},
        gauge = {
            "axis"      : {"range": [0, 100], "tickcolor": "#fff",
                           "tickfont": {"color": "#fff"}},
            "bar"       : {"color": color, "thickness": 0.25},
            "bgcolor"   : "#1e1e2e",
            "bordercolor": "#30363d",
            "steps"     : [
                {"range": [0,  35], "color": "#2d0a0a"},
                {"range": [35, 50], "color": "#2d1a00"},
                {"range": [50, 65], "color": "#2d2d00"},
                {"range": [65, 80], "color": "#003d1a"},
                {"range": [80,100], "color": "#004d26"},
            ],
            "threshold" : {"line": {"color": color, "width": 4},
                           "thickness": 0.85, "value": score}
        }
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="#0d1117",
        font_color="#fff",
        margin=dict(l=30, r=30, t=30, b=10)
    )
    return fig


def render_section_bars(section_scores):
    fig = go.Figure()
    sections = list(section_scores.keys())
    earned   = [section_scores[s]["earned"] for s in sections]
    maxes    = [section_scores[s]["max"]    for s in sections]
    pcts     = [e/m*100 for e, m in zip(earned, maxes)]

    fig.add_trace(go.Bar(
        x=pcts, y=sections, orientation='h',
        marker_color=['#00ff88' if p >= 70 else '#ffff00' if p >= 50 else '#ff4444' for p in pcts],
        text=[f"{e}/{m} pts" for e, m in zip(earned, maxes)],
        textposition='inside',
        insidetextanchor='middle'
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Score %", tickcolor="#fff",
                   gridcolor="#222", color="#fff"),
        yaxis=dict(tickfont=dict(size=13, color="#fff")),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#fff",
        height=260,
        margin=dict(l=10, r=20, t=10, b=30)
    )
    return fig


# ══════════════════════════════════════════════
# AUTH PAGES
# ══════════════════════════════════════════════
def show_login():
    st.markdown('<p class="main-header">📈 AI Stock Predictor</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">🔐 Login</p>', unsafe_allow_html=True)
        with st.form("login_form"):
            username  = st.text_input("👤 Username", placeholder="Enter your username")
            password  = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            success, msg, user = login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success(msg); st.rerun()
            else:
                st.error(msg)
        st.markdown("<br><center>Don't have an account?</center>", unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True):
            st.session_state.auth_page = "register"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def show_register():
    st.markdown('<p class="main-header">📈 AI Stock Predictor</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">🚀 Create Account</p>', unsafe_allow_html=True)
        with st.form("register_form"):
            username  = st.text_input("👤 Username", placeholder="Choose a username")
            email     = st.text_input("📧 Email",    placeholder="your@email.com")
            password  = st.text_input("🔑 Password", type="password", placeholder="Min. 6 characters")
            confirm   = st.text_input("🔑 Confirm",  type="password", placeholder="Repeat password")
            submitted = st.form_submit_button("Register", use_container_width=True)
        if submitted:
            if password != confirm: st.error("Passwords do not match.")
            else:
                success, msg = register_user(username, email, password)
                if success:
                    st.success(msg); st.session_state.auth_page = "login"; st.rerun()
                else: st.error(msg)
        st.markdown("<br><center>Already have an account?</center>", unsafe_allow_html=True)
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.auth_page = "login"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════
def show_dashboard():
    user = st.session_state.user

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div class="user-badge">👤 <b>{user["username"]}</b><br>'
            f'<small>{user["email"]}</small></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        stock_symbol = st.text_input("Stock Symbol", value="AAPL")
        period  = st.selectbox("Historical Period", ["1y","2y","3y","5y"], index=1)
        seq_len = st.slider("LSTM Sequence Length", 30, 120, 60, 10)

        st.markdown("---")
        st.markdown("## 👤 Investor Type")
        investor_type = st.radio(
            "I am a:",
            ["New Investor", "Existing Investor"],
            help="New = planning to buy | Existing = already owns this stock"
        )
        if investor_type == "New Investor":
            st.markdown("""
            <div style='background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:0.8rem;font-size:0.8rem;margin-top:0.5rem'>
            🟢 <b>BUY</b> — Good time to enter<br>
            🟡 <b>HOLD</b> — Wait for better opportunity<br>
            🔴 <b>NOT BUY</b> — Avoid entering now
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:0.8rem;font-size:0.8rem;margin-top:0.5rem'>
            🟢 <b>BUY</b> — Add more shares<br>
            🟡 <b>HOLD</b> — Keep current position<br>
            🔴 <b>SELL</b> — Exit to avoid loss
            </div>""", unsafe_allow_html=True)

        run_btn = st.button("🚀 Run Analysis", use_container_width=True)
        st.markdown("---")
        st.markdown("**Models**")
        st.markdown("- 🧠 CNN + BiLSTM — Prediction")
        st.markdown("- 🤗 FinBERT — Sentiment")
        st.markdown("- 🎯 Trust Engine — Confidence")
        st.markdown("---")
        st.markdown("**🕘 Search History**")
        for h in get_search_history(user["id"]):
            st.markdown(
                f'<div class="history-item">📊 <b>{h["symbol"]}</b> · <small>{h["searched_at"]}</small></div>',
                unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    # ── Header ───────────────────────────────
    st.markdown('<p class="main-header">📈 AI Stock Predictor</p>', unsafe_allow_html=True)
    st.markdown("CNN + BiLSTM price forecasting · FinBERT sentiment · Trust Score engine")

    if not run_btn:
        st.info("👈 Enter a stock symbol and click **Run Analysis** to begin.")
        return

    company = stock_symbol.split('.')[0].upper()

    with st.spinner(f"Analysing **{stock_symbol}** — LSTM training + FinBERT + Trust Score…"):
        save_search(user["id"], stock_symbol)

        # 1. LSTM
        try:
            predicted_price, hist_data, metrics, y_actual, y_pred_list = \
                train_and_predict_lstm(stock_symbol, sequence_length=seq_len)
        except Exception as e:
            st.error(f"❌ LSTM Error: {e}"); return

        # 2. Live Price
        live_price = get_live_price(stock_symbol)
        if live_price is None:
            live_price = float(hist_data['Close'].iloc[-1])
            st.warning("⚠️ Using last close price.")

        # 3. News + FinBERT
        news = get_stock_news(company)
        if not news: news = ["Market shows mixed response today"]
        sentiment_label, sentiment_score, sentiment_detail = analyze_sentiment(news)

        # 3b. Insider Sentiment from Finnhub
        try:
            insider = get_insider_sentiment(stock_symbol)
        except Exception:
            insider = "Neutral"

        # 4. Recommendation
        recommendation, growth_pct = get_recommendation(
            predicted_price, live_price, sentiment_label, insider, investor_type
        )

        # 5. Currency symbol
        currency = get_currency(stock_symbol)

        # 6. Confidence Score
        confidence = calculate_confidence_score(
            metrics, sentiment_label, sentiment_score,
            sentiment_detail, predicted_price, live_price, hist_data
        )

    # ══════════════════════════════════════════
    # CONFIDENCE / TRUST SCORE — Hero Section
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🎯 Overall Confidence / Trust Score")

    col_gauge, col_info = st.columns([1, 1.5])

    with col_gauge:
        st.plotly_chart(render_gauge(confidence["score"], confidence["color"]),
                        use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class="confidence-box" style="background:#161b22; border:2px solid {confidence['color']};">
            <div style="font-size:3.5rem; font-weight:900; color:{confidence['color']};">
                {confidence['emoji']} {confidence['score']}/100
            </div>
            <div style="font-size:1.6rem; font-weight:700; color:{confidence['color']}; margin:0.5rem 0;">
                {confidence['label']} CONFIDENCE
            </div>
            <div style="font-size:1rem; color:#9ca3af; margin-top:0.8rem; line-height:1.6;">
                {confidence['advice']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Section score breakdown bars
        st.markdown("#### Score Breakdown")
        st.plotly_chart(render_section_bars(confidence["section_scores"]),
                        use_container_width=True)

    # Detailed signal table
    with st.expander("🔍 View Detailed Signal Breakdown", expanded=False):
        rows = []
        for signal, vals in confidence["breakdown"].items():
            pct = vals['score'] / vals['max'] * 100
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            rows.append({
                "Signal"    : signal,
                "Score"     : f"{vals['score']}/{vals['max']}",
                "Value"     : vals['value'],
                "Rating"    : f"{bar} {pct:.0f}%"
            })
        df_signals = pd.DataFrame(rows)

        def color_score(val):
            try:
                earned, total = map(int, val.split('/'))
                pct = earned / total * 100
                if pct >= 70: return "background-color:#004d26;color:#00ff88"
                if pct >= 50: return "background-color:#3d3d00;color:#ffff00"
                return "background-color:#4d0000;color:#ff4444"
            except: return ""

        st.dataframe(
            df_signals.style.applymap(color_score, subset=["Score"]),
            use_container_width=True, hide_index=True
        )

    # ══════════════════════════════════════════
    # TOP METRICS
    # ══════════════════════════════════════════
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🏷️ Live Price",        f"{currency}{live_price:,.2f}")
    m2.metric("🔮 Predicted",          f"{currency}{predicted_price:,.2f}", f"{growth_pct:+.2f}%")
    m3.metric("📰 Sentiment",          sentiment_label, f"Score: {sentiment_score:+.3f}")
    insider_icon = "🟢" if insider=="Bullish" else "🔴" if insider=="Bearish" else "⚪"
    m4.metric("🏦 Insider Activity",   f"{insider_icon} {insider}")
    m5.metric("📊 R²",                 f"{metrics['R2']:.4f}")
    m6.metric("🎯 Direction Accuracy", f"{metrics['Direction Accuracy (%)']:.1f}%")

    # ══════════════════════════════════════════
    # RECOMMENDATION
    # ══════════════════════════════════════════
    st.markdown("### 🤖 AI Recommendation")

    # Investor type badge
    badge_color = "#00d2ff" if investor_type == "New Investor" else "#a78bfa"
    st.markdown(
        f'<div style="display:inline-block;background:{badge_color}22;'
        f'border:1px solid {badge_color};border-radius:20px;'
        f'padding:0.3rem 1rem;font-size:0.85rem;color:{badge_color};margin-bottom:0.8rem">'
        f'👤 {investor_type}</div>',
        unsafe_allow_html=True
    )

    reco_class = recommendation.replace(" ", "-")
    trust_note = f"Trust: {confidence['score']}/100 — {confidence['label']} | Insider: {insider}"

    # Context message based on investor type
    if investor_type == "New Investor":
        context = {
            "BUY"     : "✅ Good opportunity to enter this stock",
            "HOLD"    : "⏳ Wait — market not clear enough to enter",
            "NOT BUY" : "❌ Avoid entering — market signals are bearish"
        }.get(recommendation, "")
    else:
        context = {
            "BUY"  : "✅ Consider adding more shares to your position",
            "HOLD" : "⏳ Keep your current shares — no strong signal to change",
            "SELL" : "❌ Consider exiting — price expected to fall"
        }.get(recommendation, "")

    st.markdown(
        f'<div class="recommendation-box reco-{reco_class}">'
        f'{recommendation}<br>'
        f'<span style="font-size:1rem;font-weight:400">{context}</span><br>'
        f'<span style="font-size:0.8rem;opacity:0.6">{growth_pct:+.2f}% expected | {trust_note}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════
    # ACCURACY DASHBOARD
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🏆 Model Accuracy Dashboard")
    a1, a2, a3, a4, a5 = st.columns(5)
    def acc_card(col, title, value, note, color="#fff"):
        col.markdown(f"**{title}**")
        col.markdown(f'<div class="accuracy-bar"><h3 style="color:{color}">{value}</h3>'
                     f'<small>{note}</small></div>', unsafe_allow_html=True)

    acc_card(a1, "MAE",  f"{currency}{metrics['MAE']:,.2f}",  "Lower is better")
    acc_card(a2, "RMSE", f"{currency}{metrics['RMSE']:,.2f}", "Lower is better")
    r2c = "#00ff88" if metrics["R2"]>0.90 else "#ffff00" if metrics["R2"]>0.75 else "#ff4444"
    acc_card(a3, "R² Score",   f"{metrics['R2']:.4f}",   "→ 1 is best", r2c)
    dac = "#00ff88" if metrics["Direction Accuracy (%)"]>62 else "#ffff00" if metrics["Direction Accuracy (%)"]>55 else "#ff4444"
    acc_card(a4, "Direction",  f"{metrics['Direction Accuracy (%)']:.1f}%", "Up/Down calls", dac)
    mpc = "#00ff88" if metrics.get("MAPE (%)",99)<2.5 else "#ffff00" if metrics.get("MAPE (%)",99)<5 else "#ff4444"
    acc_card(a5, "MAPE",       f"{metrics.get('MAPE (%)',0):.2f}%", "% price error", mpc)

    # Actual vs Predicted
    st.markdown("#### 📉 Actual vs Predicted (Test Set)")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=y_actual,    name="Actual",    line=dict(color="#00d2ff", width=2)))
    fig_acc.add_trace(go.Scatter(y=y_pred_list, name="Predicted", line=dict(color="#ff6b6b", width=2, dash="dot")))
    fig_acc.update_layout(xaxis_title="Days", yaxis_title="Price",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
        margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_acc, use_container_width=True)

    # Residuals
    residuals = np.array(y_actual) - np.array(y_pred_list)
    fig_resid = px.histogram(residuals, nbins=40, title="Prediction Residuals",
        labels={"value":"Error"}, color_discrete_sequence=["#3a7bd5"])
    fig_resid.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff")
    st.plotly_chart(fig_resid, use_container_width=True)

    # ══════════════════════════════════════════
    # HISTORICAL CHART
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📊 Historical Price Chart")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_data.index, y=hist_data['Close'].values.flatten(),
        name="Close", line=dict(color="#00d2ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,210,255,0.07)"
    ))
    fig_hist.add_hline(y=live_price,      line_dash="dash", line_color="#ffff00",
                       annotation_text=f"Live {currency}{live_price}",      annotation_position="top left")
    fig_hist.add_hline(y=predicted_price, line_dash="dot",  line_color="#00ff88",
                       annotation_text=f"Predicted {currency}{predicted_price}", annotation_position="bottom right")
    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#fff",
        margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # ══════════════════════════════════════════
    # FINBERT SENTIMENT
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🤗 FinBERT Sentiment Analysis")
    pos = sum(1 for d in sentiment_detail if d['label']=='positive')
    neg = sum(1 for d in sentiment_detail if d['label']=='negative')
    neu = sum(1 for d in sentiment_detail if d['label']=='neutral')
    s1, s2, s3 = st.columns(3)
    s1.metric("✅ Positive", pos)
    s2.metric("❌ Negative", neg)
    s3.metric("➖ Neutral",  neu)

    fig_sent = go.Figure(go.Pie(
        labels=["Positive","Negative","Neutral"], values=[pos,neg,neu], hole=0.55,
        marker_colors=["#00ff88","#ff4444","#aaaaaa"]))
    fig_sent.update_layout(paper_bgcolor="#0d1117", font_color="#fff",
                           margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown("#### 📰 Headlines")
    if sentiment_detail:
        df_news = pd.DataFrame(sentiment_detail)
        df_news.columns = ["Headline","Sentiment","Confidence"]
        def color_label(val):
            if val=="positive": return "background-color:#004d26;color:#00ff88"
            if val=="negative": return "background-color:#4d0000;color:#ff4444"
            return "background-color:#333;color:#fff"
        st.dataframe(df_news.style.applymap(color_label, subset=["Sentiment"]),
                     use_container_width=True, height=360)

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ══════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════
if st.session_state.logged_in:
    show_dashboard()
else:
    if st.session_state.auth_page == "register":
        show_register()
    else:
        show_login()