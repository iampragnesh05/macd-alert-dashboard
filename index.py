"""
MACD Zero-Line Alert Dashboard
================================
Auto-refreshing candlestick chart with a MACD indicator underneath.
Fires an alert whenever the MACD line crosses above (bullish) or
below (bearish) the zero line.

RUN THIS:
    pip install streamlit yfinance pandas plotly
    streamlit run macd_alert_app.py

DATA SOURCE:
    Right now it uses yfinance (free, no account, ~15min delayed) so you
    can test everything end-to-end immediately. When you're ready to plug
    in a real broker feed (Upstox / Angel One / Fyers), you only need to
    replace the `fetch_candles()` function below — everything else
    (MACD math, alert logic, charting) stays exactly the same, because
    it all just expects a DataFrame with open/high/low/close columns.
"""

import time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------
# 1. DATA FETCHING  -- swap this function out later for a live broker feed
# ----------------------------------------------------------------------
def fetch_candles(symbol: str, interval: str, period: str = "5d") -> pd.DataFrame:
    """
    Returns a DataFrame indexed by time with columns: open, high, low, close, volume

    symbol examples for NSE via yfinance: 'RELIANCE.NS', 'TCS.NS', 'INFY.NS'
    interval examples: '1m', '5m', '15m', '1d'
    """
    data = yf.download(symbol, interval=interval, period=period, progress=False)
    if data.empty:
        return data
    # yfinance sometimes returns multi-index columns -- flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data = data.rename(columns=str.lower)
    return data[["open", "high", "low", "close", "volume"]]


def get_current_price(symbol: str, df: pd.DataFrame) -> dict:
    """
    Returns the latest price plus change vs previous close, along with
    the timestamp that price is "as of" and how much lag that implies
    vs right now -- so you can see if your data is fresh or delayed.
    """
    price, prev_close = None, None
    try:
        info = yf.Ticker(symbol).fast_info
        price = info.get("last_price")
        prev_close = info.get("previous_close") or info.get("regular_market_previous_close")
    except Exception:
        pass

    if price is None and not df.empty:
        price = df["close"].iloc[-1]
    if prev_close is None and len(df) > 1:
        prev_close = df["close"].iloc[-2]
    elif prev_close is None and not df.empty:
        prev_close = df["close"].iloc[-1]

    # "as_of" = timestamp of the latest candle we have -- the most honest
    # way to know how fresh the price actually is (yfinance doesn't give
    # a separate timestamp for fast_info's live quote).
    as_of = df.index[-1] if not df.empty else None
    lag_seconds = None
    if as_of is not None:
        now = pd.Timestamp.now(tz=as_of.tz) if as_of.tzinfo is not None else pd.Timestamp.now()
        lag_seconds = max((now - as_of).total_seconds(), 0)

    change = (price - prev_close) if (price is not None and prev_close is not None) else 0
    pct_change = (change / prev_close * 100) if prev_close else 0
    return {
        "price": price, "change": change, "pct_change": pct_change,
        "as_of": as_of, "lag_seconds": lag_seconds,
    }


# ----------------------------------------------------------------------
# 2. INDICATOR MATH -- MACD (12, 26, 9 by default, fully adjustable)
# ----------------------------------------------------------------------
def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9, price_col="close") -> pd.DataFrame:
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["histogram"] = df["macd"] - df["signal"]
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range -- used to size stop-loss/target distances that flex with volatility."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_filter_columns(df: pd.DataFrame, atr_period: int = 14, vol_period: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = compute_atr(df, atr_period)
    df["vol_avg"] = df["volume"].rolling(vol_period).mean()
    return df


# ----------------------------------------------------------------------
# 3. ALERT LOGIC -- fires when MACD line crosses the zero line,
#    then passes through 5 optional quality filters before it counts
#    as a real alert. Each filter can be switched off independently.
# ----------------------------------------------------------------------
def detect_zero_cross(
    df: pd.DataFrame,
    confirm_candles: int = 2,          # 1. confirmation filter
    require_volume: bool = True,       # 2. volume confirmation
    volume_mult: float = 1.2,
    min_strength: float = 0.0,         # 3. crossover strength
    require_signal_agreement: bool = True,  # 5. signal-line agreement
    calc_risk_levels: bool = True,     # 4. risk levels (stop/target)
    atr_mult_stop: float = 1.5,
    risk_reward: float = 2.0,
) -> list[dict]:
    alerts = []
    macd, signal = df["macd"], df["signal"]

    for i in range(1, len(df)):
        prev, curr = macd.iloc[i - 1], macd.iloc[i]
        if pd.isna(prev) or pd.isna(curr):
            continue

        direction = None
        if prev < 0 and curr >= 0:
            direction = "BULLISH"
        elif prev > 0 and curr <= 0:
            direction = "BEARISH"
        if direction is None:
            continue

        # --- Filter 1: confirmation -- MACD must hold the new side for N candles ---
        if confirm_candles > 0:
            end = i + confirm_candles
            if end >= len(df):
                continue  # too recent to confirm yet -- will re-check on next refresh
            future = macd.iloc[i:end + 1]
            if direction == "BULLISH" and not (future >= 0).all():
                continue
            if direction == "BEARISH" and not (future <= 0).all():
                continue

        # --- Filter 2: volume confirmation ---
        if require_volume:
            vol, vol_avg = df["volume"].iloc[i], df["vol_avg"].iloc[i]
            if pd.isna(vol_avg) or vol < vol_avg * volume_mult:
                continue

        # --- Filter 3: crossover strength -- ignore too-shallow crosses ---
        if min_strength > 0 and abs(curr) < min_strength:
            continue

        # --- Filter 5: signal-line agreement ---
        if require_signal_agreement:
            sig = signal.iloc[i]
            if direction == "BULLISH" and not (curr > sig):
                continue
            if direction == "BEARISH" and not (curr < sig):
                continue

        # --- Filter 4: risk levels via ATR ---
        entry = df["close"].iloc[i]
        stop = target = None
        if calc_risk_levels:
            atr = df["atr"].iloc[i]
            if not pd.isna(atr):
                risk_dist = atr_mult_stop * atr
                if direction == "BULLISH":
                    stop, target = entry - risk_dist, entry + risk_dist * risk_reward
                else:
                    stop, target = entry + risk_dist, entry - risk_dist * risk_reward

        alerts.append({
            "time": df.index[i], "type": direction,
            "message": f"MACD crossed {'ABOVE' if direction == 'BULLISH' else 'BELOW'} zero (confirmed)",
            "macd": round(curr, 4),
            "entry": round(entry, 2),
            "stop": round(stop, 2) if stop is not None else None,
            "target": round(target, 2) if target is not None else None,
        })
    return alerts


# ----------------------------------------------------------------------
# 4. CHART -- candlesticks on top, MACD panel underneath, alerts marked
# ----------------------------------------------------------------------
def build_chart(df: pd.DataFrame, alerts: list[dict], symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
        vertical_spacing=0.03, subplot_titles=(symbol, "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#2962FF")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["signal"], name="Signal", line=dict(color="#FF6D00")), row=2, col=1)
    colors = np.where(df["histogram"] >= 0, "#26A69A", "#EF5350")
    fig.add_trace(go.Bar(x=df.index, y=df["histogram"], name="Histogram", marker_color=colors), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    if alerts:
        bull = [a for a in alerts if a["type"] == "BULLISH"]
        bear = [a for a in alerts if a["type"] == "BEARISH"]
        if bull:
            fig.add_trace(go.Scatter(
                x=[a["time"] for a in bull], y=[a["macd"] for a in bull],
                mode="markers", marker=dict(symbol="triangle-up", size=12, color="#00C853"),
                name="Bullish cross",
            ), row=2, col=1)
        if bear:
            fig.add_trace(go.Scatter(
                x=[a["time"] for a in bear], y=[a["macd"] for a in bear],
                mode="markers", marker=dict(symbol="triangle-down", size=12, color="#D50000"),
                name="Bearish cross",
            ), row=2, col=1)

        # Draw stop/target for the most recent alert on the price panel
        last = alerts[-1]
        if last.get("stop") is not None and last.get("target") is not None:
            fig.add_hline(y=last["stop"], line_dash="dash", line_color="#D50000",
                           annotation_text=f"Stop {last['stop']}", annotation_position="right",
                           row=1, col=1)
            fig.add_hline(y=last["target"], line_dash="dash", line_color="#00C853",
                           annotation_text=f"Target {last['target']}", annotation_position="right",
                           row=1, col=1)
            fig.add_hline(y=last["entry"], line_dash="dot", line_color="gray",
                           annotation_text=f"Entry {last['entry']}", annotation_position="right",
                           row=1, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ----------------------------------------------------------------------
# 5. STREAMLIT APP
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MACD Zero-Line Alerts", layout="wide")
    st.title("📈 MACD Zero-Line Alert Dashboard")

    with st.sidebar:
        st.header("Settings")
        symbol = st.text_input("Symbol (yfinance format)", value="RELIANCE.NS",
                                help="NSE stocks: TICKER.NS, e.g. TCS.NS, INFY.NS, HDFCBANK.NS")
        interval = st.selectbox("Candle interval", ["1m", "5m", "15m", "1d"], index=1)
        period = st.selectbox("Lookback period", ["1d", "5d", "1mo"], index=1)
        fast = st.number_input("Fast EMA", value=12, min_value=2)
        slow = st.number_input("Slow EMA", value=26, min_value=2)
        signal = st.number_input("Signal EMA", value=9, min_value=2)

        with st.expander("🔧 Alert filters", expanded=True):
            confirm_candles = st.slider(
                "1. Confirmation candles", 0, 5, 2,
                help="MACD must stay on the new side for this many candles before the alert fires. 0 = off (fires instantly, more false alerts).")
            require_volume = st.checkbox("2. Require volume confirmation", value=True)
            volume_mult = st.slider("   Volume must be >= average x", 1.0, 3.0, 1.2, 0.1,
                                     disabled=not require_volume)
            min_strength = st.number_input(
                "3. Minimum crossover strength (MACD abs value)", value=0.0, min_value=0.0, step=0.1,
                help="Ignore crosses where MACD barely nudges past zero. 0 = off.")
            require_signal_agreement = st.checkbox(
                "5. Require signal-line agreement", value=True,
                help="Bullish alert only if MACD is also above its Signal line (and vice versa for bearish).")
            calc_risk_levels = st.checkbox("4. Show risk levels (stop/target)", value=True)
            atr_mult_stop = st.slider("   Stop distance (x ATR)", 0.5, 3.0, 1.5, 0.1, disabled=not calc_risk_levels)
            risk_reward = st.slider("   Risk:Reward ratio", 1.0, 4.0, 2.0, 0.5, disabled=not calc_risk_levels)

        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
        refresh_now = st.button("Refresh now")

    df = fetch_candles(symbol, interval, period)
    if df.empty:
        st.error("No data returned. Check the symbol / interval combination.")
        return

    df = compute_macd(df, fast=fast, slow=slow, signal=signal)
    df = add_filter_columns(df)
    alerts = detect_zero_cross(
        df,
        confirm_candles=confirm_candles,
        require_volume=require_volume,
        volume_mult=volume_mult,
        min_strength=min_strength,
        require_signal_agreement=require_signal_agreement,
        calc_risk_levels=calc_risk_levels,
        atr_mult_stop=atr_mult_stop,
        risk_reward=risk_reward,
    )

    quote = get_current_price(symbol, df)
    price_col, macd_col, state_col = st.columns(3)
    price_col.metric(
        f"{symbol} — Current Price",
        f"₹{quote['price']:.2f}" if quote["price"] is not None else "N/A",
        f"{quote['change']:+.2f} ({quote['pct_change']:+.2f}%)",
    )
    if quote["as_of"] is not None:
        lag = quote["lag_seconds"]
        if lag is not None and lag < 120:
            lag_text = f"{int(lag)} sec ago"
        elif lag is not None:
            lag_text = f"{int(lag // 60)} min ago"
        else:
            lag_text = ""
        price_col.caption(f"🕒 As of {quote['as_of'].strftime('%d %b, %H:%M:%S')} ({lag_text})")
    latest_macd = df["macd"].iloc[-1]
    macd_col.metric("Latest MACD", f"{latest_macd:.4f}")
    state_col.metric("Zone", "🟢 Above zero (bullish)" if latest_macd >= 0 else "🔴 Below zero (bearish)")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(build_chart(df, alerts, symbol), use_container_width=True)
    with col2:
        st.subheader("🔔 Alerts")
        if not alerts:
            st.info("No zero-line crossovers in this window yet.")
        else:
            for a in reversed(alerts[-15:]):
                icon = "🟢" if a["type"] == "BULLISH" else "🔴"
                st.write(f"{icon} **{a['type']}** — {a['time'].strftime('%Y-%m-%d %H:%M')}")
                st.caption(f"{a['message']} (MACD={a['macd']})")
                if a.get("stop") is not None:
                    st.caption(f"Entry ₹{a['entry']} · Stop ₹{a['stop']} · Target ₹{a['target']}")

    if auto_refresh:
        time.sleep(30)
        st.rerun()
    if refresh_now:
        st.rerun()


if __name__ == "__main__":
    main()