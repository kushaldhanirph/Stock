"""
Buffett Style Value Investing Screener — Indian Market (NSE)
--------------------------------------------------------------
Fetches live fundamental data for NSE-listed stocks via yfinance,
scores each stock using Warren Buffett style value-investing
criteria, estimates intrinsic value (Graham Number), and ranks
stocks by "Margin of Safety".

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free:
    Push this folder to a public GitHub repo, then deploy on
    https://share.streamlit.io  (Streamlit Community Cloud).
"""

import math
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Buffett Value Screener — India",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------
# Default watchlist — a broad basket of large & mid cap NSE stocks.
# Users can edit this list from the sidebar.
# ----------------------------------------------------------------------
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "ASIANPAINT.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS",
    "ONGC.NS", "COALINDIA.NS", "TECHM.NS", "ADANIPORTS.NS", "GRASIM.NS",
    "BAJAJFINSV.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BPCL.NS", "IOC.NS",
    "PIDILITIND.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "COLPAL.NS",
    "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
]

# ----------------------------------------------------------------------
# Scoring logic — Buffett style criteria
# ----------------------------------------------------------------------
def score_stock(info: dict):
    """Return (score out of 100, list of (label, verdict, detail) rows)."""
    score = 0
    max_score = 0
    rows = []

    roe = info.get("returnOnEquity")
    if roe is not None:
        max_score += 20
        pts = 20 if roe > 0.20 else 15 if roe > 0.15 else 8 if roe > 0.10 else 0
        score += pts
        rows.append(("ROE (Return on Equity)", f"{roe*100:.1f}%", pts, 20))

    de = info.get("debtToEquity")
    if de is not None:
        de_ratio = de / 100  # yfinance reports as percentage
        max_score += 20
        pts = 20 if de_ratio < 0.3 else 12 if de_ratio < 0.6 else 5 if de_ratio < 1.0 else 0
        score += pts
        rows.append(("Debt / Equity", f"{de_ratio:.2f}", pts, 20))

    pe = info.get("trailingPE")
    if pe is not None and pe > 0:
        max_score += 15
        pts = 15 if pe < 15 else 8 if pe < 25 else 3 if pe < 35 else 0
        score += pts
        rows.append(("P/E Ratio", f"{pe:.1f}", pts, 15))

    pb = info.get("priceToBook")
    if pb is not None and pb > 0:
        max_score += 15
        pts = 15 if pb < 1.5 else 8 if pb < 3 else 3 if pb < 5 else 0
        score += pts
        rows.append(("P/B Ratio", f"{pb:.2f}", pts, 15))

    fcf = info.get("freeCashflow")
    if fcf is not None:
        max_score += 15
        pts = 15 if fcf > 0 else 0
        score += pts
        rows.append(("Free Cash Flow", f"₹{fcf/1e7:,.0f} Cr", pts, 15))

    eg = info.get("earningsGrowth")
    if eg is not None:
        max_score += 15
        pts = 15 if eg > 0.15 else 8 if eg > 0.05 else 3 if eg > 0 else 0
        score += pts
        rows.append(("Earnings Growth (YoY)", f"{eg*100:.1f}%", pts, 15))

    final = round((score / max_score) * 100, 1) if max_score > 0 else 0.0
    return final, rows


def graham_number(info: dict):
    """Classic Graham Number intrinsic value estimate:
    sqrt(22.5 * EPS * Book Value per Share)
    Returns None if inputs unavailable or negative.
    """
    eps = info.get("trailingEps")
    bvps = info.get("bookValue")
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # refresh every 12 hours
def fetch_data(tickers: tuple):
    records = []
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                continue
            score, rows = score_stock(info)
            gnum = graham_number(info)
            mos = round((gnum - price) / gnum * 100, 1) if gnum else None
            records.append({
                "Ticker": t.replace(".NS", ""),
                "Name": info.get("shortName", t),
                "Sector": info.get("sector", "N/A"),
                "Price (₹)": round(price, 2),
                "Buffett Score": score,
                "Graham Value (₹)": round(gnum, 2) if gnum else None,
                "Margin of Safety (%)": mos,
                "ROE (%)": round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
                "D/E": round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") is not None else None,
                "P/E": round(info.get("trailingPE"), 1) if info.get("trailingPE") else None,
                "P/B": round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
                "_rows": rows,
            })
        except Exception:
            continue
        time.sleep(0.15)  # be gentle with the free data API
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

ticker_text = st.sidebar.text_area(
    "NSE Tickers (comma separated, use .NS suffix)",
    value=", ".join(DEFAULT_TICKERS),
    height=150,
)
tickers = tuple(sorted(set(t.strip().upper() for t in ticker_text.split(",") if t.strip())))

min_score = st.sidebar.slider("Minimum Buffett Score", 0, 100, 50)
only_undervalued = st.sidebar.checkbox("শুধু Margin of Safety > 0 দেখাও (undervalued only)", value=False)

if st.sidebar.button("🔄 Force refresh data now"):
    st.cache_data.clear()

st.sidebar.caption(
    "ডেটা প্রতি ১২ ঘণ্টায় স্বয়ংক্রিয়ভাবে refresh হয় (cache TTL)। "
    "সোর্স: Yahoo Finance (yfinance), সম্পূর্ণ ফ্রি।"
)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("📈 বাফেট-স্টাইল ভ্যালু স্টক স্ক্রিনার — ভারতীয় বাজার (NSE)")
st.caption(
    "Warren Buffett এর ভ্যালু ইনভেস্টিং নীতি (উচ্চ ROE, কম ঋণ, যুক্তিসঙ্গত P/E ও P/B, "
    "পজিটিভ Free Cash Flow ও Earnings Growth) অনুযায়ী স্টক স্ক্রিন ও র‍্যাঙ্ক করে।"
)

with st.spinner("লাইভ ডেটা আনা হচ্ছে... (প্রথমবার একটু সময় লাগতে পারে)"):
    df = fetch_data(tickers)

if df.empty:
    st.error("কোনো ডেটা পাওয়া যায়নি। টিকার লিস্ট চেক করুন অথবা কিছুক্ষণ পর আবার চেষ্টা করুন।")
    st.stop()

filtered = df[df["Buffett Score"] >= min_score]
if only_undervalued:
    filtered = filtered[filtered["Margin of Safety (%)"] > 0]
filtered = filtered.sort_values("Buffett Score", ascending=False)

st.markdown(f"**{len(filtered)} / {len(df)}** টি স্টক আপনার ফিল্টার পাস করেছে।")

display_cols = [
    "Ticker", "Name", "Sector", "Price (₹)", "Buffett Score",
    "Graham Value (₹)", "Margin of Safety (%)", "ROE (%)", "D/E", "P/E", "P/B",
]
st.dataframe(
    filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=480,
)

st.markdown("### 🔍 বিস্তারিত স্কোর ব্রেকডাউন")
pick = st.selectbox("একটা স্টক বেছে নিন বিস্তারিত দেখতে", filtered["Ticker"] if not filtered.empty else df["Ticker"])
row = df[df["Ticker"] == pick].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Buffett Score", f"{row['Buffett Score']}/100")
col2.metric("Current Price", f"₹{row['Price (₹)']}")
mos_val = row["Margin of Safety (%)"]
col3.metric("Margin of Safety", f"{mos_val}%" if mos_val is not None else "N/A")

detail_df = pd.DataFrame(row["_rows"], columns=["Criterion", "Value", "Points", "Max"])
st.table(detail_df)

st.caption(
    f"সর্বশেষ ডেটা আপডেট (cache): {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    "⚠️ এটি শিক্ষামূলক টুল, বিনিয়োগ পরামর্শ নয়। কেনার আগে নিজে গবেষণা করুন বা SEBI-নিবন্ধিত উপদেষ্টার পরামর্শ নিন।"
)
