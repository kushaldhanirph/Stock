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

import io
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
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
# NSE publishes official index-constituent CSVs (Nifty 50/100/200/500 etc).
# We fetch these live so the screener covers the REAL, current universe
# instead of a small hardcoded list. NSE blocks plain requests without
# browser-like headers, so we open the homepage first to collect cookies.
# ----------------------------------------------------------------------
NSE_INDEX_CSV = {
    "Nifty 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "Nifty 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # index membership changes rarely
def fetch_index_constituents(universe: str):
    """Download the official NSE constituent list for a given index and
    return a sorted list of tickers like 'RELIANCE.NS'. Returns None on
    failure so the caller can fall back to the bundled list."""
    url = NSE_INDEX_CSV.get(universe)
    if not url:
        return None
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=10)  # sets cookies
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].dropna().astype(str).str.strip()
        return sorted(f"{s}.NS" for s in symbols if s)
    except Exception:
        return None


NSE_FULL_LIST_URL = "https://archives.nseindia.com/content/equity/EQUITY_L.csv"


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_full_nse_list():
    """Download NSE's complete list of listed equities (Series = EQ), the
    closest free equivalent to 'every NSE stock' (~2000 names). Returns
    None on failure so the caller can fall back to the bundled list.
    Note: BSE does not publish an equally stable free full-list feed
    (its old Bhavcopy format was discontinued in July 2024 and replaced
    with a link that changes daily), so a reliable 'all BSE stocks'
    option isn't offered here — see the README for details.
    """
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=10)
        resp = session.get(NSE_FULL_LIST_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().upper() for c in df.columns]
        if "SERIES" in df.columns:
            df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        symbols = df["SYMBOL"].dropna().astype(str).str.strip()
        return sorted(f"{s}.NS" for s in symbols if s)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Fallback watchlist — used only if the live NSE fetch above fails
# (e.g. NSE temporarily blocking the request). Users can also switch to
# "Custom" in the sidebar and paste/edit any list, including BSE tickers
# with a ".BO" suffix (e.g. 500325.BO for Reliance on BSE).
# ----------------------------------------------------------------------
FALLBACK_TICKERS = [
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


# ----------------------------------------------------------------------
# Market-cap based Large / Mid / Small Cap tagging.
# NOTE: SEBI's official classification is RANK-based (Large = 1st-100th
# company by market cap, Mid = 101st-250th, Small = 251st onward) and
# the exact cutoff values are revised twice a year by AMFI. We don't
# have a free live feed of that ranking, so this uses commonly-cited
# approximate value cutoffs instead — good enough to bucket stocks, but
# a stock near a boundary may be classified slightly differently than
# the official AMFI list. Edit the two constants below if you have a
# more current cutoff figure.
# ----------------------------------------------------------------------
LARGE_CAP_MIN_CR = 20000   # ₹20,000+ Crore market cap ≈ Large Cap
MID_CAP_MIN_CR = 5000      # ₹5,000–20,000 Crore ≈ Mid Cap; below ≈ Small Cap


def market_cap_category(market_cap):
    """market_cap is in raw rupees (as yfinance returns it)."""
    if market_cap is None:
        return "N/A"
    cr = market_cap / 1e7
    if cr >= LARGE_CAP_MIN_CR:
        return "Large Cap"
    if cr >= MID_CAP_MIN_CR:
        return "Mid Cap"
    return "Small Cap"


# ----------------------------------------------------------------------
# Rule-based BUY / HOLD / SELL tag.
# This is purely a mechanical read of the Buffett Score + Margin of
# Safety this app already calculates — NOT personalized financial
# advice. Thresholds are intentionally conservative and editable below.
# ----------------------------------------------------------------------
BUY_SCORE_MIN = 65        # fundamentals strong enough to consider buying
BUY_MOS_MIN = 15          # and trading at least 15% below intrinsic value
SELL_SCORE_MAX = 45       # fundamentals weak
SELL_MOS_MAX = -25        # or trading 25%+ ABOVE intrinsic value (expensive)


def get_verdict(score: float, mos):
    """Return (label, css_color) for the Buy/Hold/Sell tag."""
    if mos is not None:
        if score >= BUY_SCORE_MIN and mos > BUY_MOS_MIN:
            return "🟢 কেনার মতো (Buy Zone)", "#1a7f37"
        if score < SELL_SCORE_MAX or mos < SELL_MOS_MAX:
            return "🔴 বিক্রি বিবেচনা করুন (Sell Zone)", "#cf222e"
        return "🟡 হোল্ড করুন (Hold Zone)", "#9a6700"
    # No reliable Graham value (e.g. negative EPS) — judge on score alone
    if score >= 70:
        return "🟢 কেনার মতো (Buy Zone)", "#1a7f37"
    if score < 40:
        return "🔴 বিক্রি বিবেচনা করুন (Sell Zone)", "#cf222e"
    return "🟡 হোল্ড করুন (Hold Zone)", "#9a6700"


def _fetch_one(t: str, retries: int = 0):
    """retries>0 adds a short pre-request delay, used for the second
    (slower, gentler) retry pass so we don't hammer Yahoo again the same way."""
    if retries:
        time.sleep(0.5 * retries + random.uniform(0, 0.5))
    try:
        info = yf.Ticker(t).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return None
        score, rows = score_stock(info)
        gnum = graham_number(info)
        mos = round((gnum - price) / gnum * 100, 1) if gnum else None
        verdict, verdict_color = get_verdict(score, mos)
        mcap = info.get("marketCap")
        return {
            "Ticker": t.replace(".NS", "").replace(".BO", ""),
            "Name": info.get("shortName", t),
            "Sector": info.get("sector", "N/A"),
            "Cap Category": market_cap_category(mcap),
            "Market Cap (₹ Cr)": round(mcap / 1e7, 0) if mcap else None,
            "Price (₹)": round(price, 2),
            "Buffett Score": score,
            "Verdict": verdict,
            "Graham Value (₹)": round(gnum, 2) if gnum else None,
            "Margin of Safety (%)": mos,
            "ROE (%)": round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
            "D/E": round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") is not None else None,
            "P/E": round(info.get("trailingPE"), 1) if info.get("trailingPE") else None,
            "P/B": round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            "_rows": rows,
            "_verdict_color": verdict_color,
        }
    except Exception:
        return None


# Bump this whenever the shape of the record dict in _fetch_one changes
# (new/renamed/removed columns). Passing it into fetch_data's cache key
# guarantees Streamlit invalidates any old cached results automatically,
# instead of silently reusing a DataFrame with a different schema.
DATA_SCHEMA_VERSION = 2


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)  # refresh every 12 hours
def fetch_data(tickers: tuple, max_workers: int = 8, schema_version: int = DATA_SCHEMA_VERSION):
    """Fetch fundamentals for all tickers in parallel. Yahoo Finance
    (yfinance's free data source) sometimes rate-limits a burst of
    parallel requests, silently failing a chunk of tickers on the first
    pass — so any that fail get a couple of gentler, slower retries
    afterwards instead of being dropped outright."""
    records = []
    total = len(tickers)
    progress = st.progress(0.0, text=f"0 / {total} স্টক প্রসেস হয়েছে...")
    done = 0

    def _run_pass(ticker_list, workers, retry_num, label):
        nonlocal done
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, t, retry_num): t for t in ticker_list}
            for fut in as_completed(futures):
                t = futures[fut]
                result = fut.result()
                if result:
                    records.append(result)
                else:
                    failed.append(t)
                done += 1
                if done % 5 == 0 or done == total:
                    progress.progress(min(done / total, 1.0), text=f"{label}: {done} স্টক প্রসেস হয়েছে...")
        return failed

    failed = _run_pass(tickers, max_workers, 0, f"0 / {total} স্টক প্রসেস হয়েছে")

    # Retry failures up to twice, slower and gentler each time, so a
    # temporary rate-limit doesn't permanently drop a stock from results.
    for retry_num in (1, 2):
        if not failed:
            break
        done = total - len(failed)  # so the progress bar reflects the retry subset
        failed = _run_pass(failed, max(2, max_workers // 2), retry_num, f"পুনরায় চেষ্টা #{retry_num}")

    progress.empty()
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

universe_choice = st.sidebar.selectbox(
    "স্টক ইউনিভার্স (কতগুলো স্টক স্ক্যান করবে)",
    ["NSE - সব স্টক (Full ~2000, ধীর)", "Nifty 500", "Nifty 200", "Nifty 100", "Nifty 50", "Custom (নিজে লিখুন)"],
    index=0,  # default: the full NSE list, since that's what most people expect
)

if universe_choice == "Custom (নিজে লিখুন)":
    ticker_text = st.sidebar.text_area(
        "Tickers (comma separated). NSE হলে .NS, BSE হলে .BO সাফিক্স দিন",
        value=", ".join(FALLBACK_TICKERS),
        height=150,
    )
    tickers = tuple(sorted(set(t.strip().upper() for t in ticker_text.split(",") if t.strip())))
    source_note = f"কাস্টম লিস্ট থেকে {len(tickers)}টি টিকার নেওয়া হয়েছে।"
elif universe_choice == "NSE - সব স্টক (Full ~2000, ধীর)":
    live_list = fetch_full_nse_list()
    if live_list:
        tickers = tuple(live_list)
        source_note = f"✅ NSE-এর সম্পূর্ণ লিস্ট আনা হয়েছে — মোট {len(tickers)}টি স্টক (Series: EQ)।"
    else:
        tickers = tuple(FALLBACK_TICKERS)
        source_note = (
            "⚠️ NSE থেকে সম্পূর্ণ লিস্ট আনা যায়নি (সাময়িক ব্লক/নেটওয়ার্ক সমস্যা), "
            f"তাই fallback লিস্ট ({len(tickers)}টি বড় স্টক) ব্যবহার করা হচ্ছে।"
        )
else:
    live_list = fetch_index_constituents(universe_choice)
    if live_list:
        tickers = tuple(live_list)
        source_note = f"✅ NSE থেকে লাইভ **{universe_choice}** লিস্ট আনা হয়েছে — মোট {len(tickers)}টি স্টক।"
    else:
        tickers = tuple(FALLBACK_TICKERS)
        source_note = (
            f"⚠️ NSE থেকে লাইভ {universe_choice} লিস্ট আনা যায়নি (সাময়িক ব্লক/নেটওয়ার্ক সমস্যা), "
            f"তাই বিল্ট-ইন fallback লিস্ট ({len(tickers)}টি বড় স্টক) ব্যবহার করা হচ্ছে। "
            "'Force refresh' চেপে আবার চেষ্টা করে দেখুন।"
        )

st.sidebar.caption(source_note)

if len(tickers) > 200:
    st.sidebar.warning(
        f"{len(tickers)}টি স্টক স্ক্যান করতে প্রথমবার বেশ কয়েক মিনিট সময় লাগতে পারে, এবং Yahoo Finance "
        "মাঝে মাঝে অতিরিক্ত রিকোয়েস্টে সাময়িক rate-limit করতে পারে (কিছু স্টক তখন স্কিপ হয়ে যাবে)। "
        "ফলাফল ১২ ঘণ্টা cache থাকবে, তাই পরের ভিজিটে সাথে সাথে দেখাবে।"
    )

min_score = st.sidebar.slider("Minimum Buffett Score", 0, 100, 50)
only_undervalued = st.sidebar.checkbox("শুধু Margin of Safety > 0 দেখাও (undervalued only)", value=False)
cap_filter = st.sidebar.multiselect(
    "Market Cap দিয়ে ফিল্টার করুন",
    ["Large Cap", "Mid Cap", "Small Cap"],
    default=["Large Cap", "Mid Cap", "Small Cap"],
    help=f"আনুমানিক ভাগ: Large Cap ≥ ₹{LARGE_CAP_MIN_CR:,} Cr, Mid Cap ₹{MID_CAP_MIN_CR:,}–{LARGE_CAP_MIN_CR:,} Cr, তার নিচে Small Cap। এটা SEBI/AMFI-এর অফিসিয়াল rank-ভিত্তিক কাট-অফ নয়, একটা কাছাকাছি হিসাব।",
)
verdict_filter = st.sidebar.multiselect(
    "Verdict দিয়ে ফিল্টার করুন",
    ["🟢 কেনার মতো (Buy Zone)", "🟡 হোল্ড করুন (Hold Zone)", "🔴 বিক্রি বিবেচনা করুন (Sell Zone)"],
    default=["🟢 কেনার মতো (Buy Zone)", "🟡 হোল্ড করুন (Hold Zone)", "🔴 বিক্রি বিবেচনা করুন (Sell Zone)"],
)

if st.sidebar.button("🔄 Force refresh data now"):
    st.cache_data.clear()

st.sidebar.caption(
    "ডেটা প্রতি ১২ ঘণ্টায় স্বয়ংক্রিয়ভাবে refresh হয় (cache TTL)। "
    "সোর্স: NSE (স্টক লিস্ট) + Yahoo Finance/yfinance (ফান্ডামেন্টাল ডেটা), সম্পূর্ণ ফ্রি।"
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

# Safety net: if a stale/partial cache ever slips through with a
# different shape than expected (e.g. right after a code update), patch
# in any missing columns instead of crashing the whole app.
_expected_cols = {
    "Cap Category": "N/A", "Market Cap (₹ Cr)": None, "Verdict": "🟡 হোল্ড করুন (Hold Zone)",
    "Graham Value (₹)": None, "Margin of Safety (%)": None, "ROE (%)": None,
    "D/E": None, "P/E": None, "P/B": None, "Sector": "N/A",
}
if not df.empty:
    missing = [c for c in _expected_cols if c not in df.columns]
    if missing:
        st.cache_data.clear()
        for c in missing:
            df[c] = _expected_cols[c]
        st.warning(
            "পুরনো cache-এ ডেটার গঠন মিলছিল না, তাই cache মুছে ফেলা হয়েছে। "
            "পেজটা একবার রিফ্রেশ (Ctrl/Cmd+R) করলে সম্পূর্ণ সঠিক ডেটা দেখাবে।"
        )

st.info(
    f"📋 এখন সিলেক্টেড ইউনিভার্স: **{universe_choice}** — লিস্টে মোট **{len(tickers)}**টি টিকার ছিল, "
    f"যার মধ্যে **{len(df)}**টি স্টকের ডেটা সফলভাবে পাওয়া গেছে। "
    "(ভিন্ন সংখ্যা চাইলে সাইডবারের 'স্টক ইউনিভার্স' ড্রপডাউন থেকে বদলে নিন।)"
)

if df.empty:
    st.error("কোনো ডেটা পাওয়া যায়নি। টিকার লিস্ট চেক করুন অথবা কিছুক্ষণ পর আবার চেষ্টা করুন।")
    st.stop()

filtered = df[df["Buffett Score"] >= min_score]
if only_undervalued:
    filtered = filtered[filtered["Margin of Safety (%)"] > 0]
if verdict_filter:
    filtered = filtered[filtered["Verdict"].isin(verdict_filter)]
if cap_filter:
    filtered = filtered[filtered["Cap Category"].isin(cap_filter)]
filtered = filtered.sort_values("Buffett Score", ascending=False)

st.markdown(f"**{len(filtered)} / {len(df)}** টি স্টক আপনার ফিল্টার পাস করেছে।")

buy_n = (df["Verdict"].str.contains("Buy")).sum()
hold_n = (df["Verdict"].str.contains("Hold")).sum()
sell_n = (df["Verdict"].str.contains("Sell")).sum()
c1, c2, c3 = st.columns(3)
c1.metric("🟢 Buy Zone", buy_n)
c2.metric("🟡 Hold Zone", hold_n)
c3.metric("🔴 Sell Zone", sell_n)

large_n = (df["Cap Category"] == "Large Cap").sum()
mid_n = (df["Cap Category"] == "Mid Cap").sum()
small_n = (df["Cap Category"] == "Small Cap").sum()
d1, d2, d3 = st.columns(3)
d1.metric("Large Cap", large_n)
d2.metric("Mid Cap", mid_n)
d3.metric("Small Cap", small_n)

display_cols = [
    "Ticker", "Name", "Sector", "Cap Category", "Market Cap (₹ Cr)", "Price (₹)",
    "Buffett Score", "Verdict", "Graham Value (₹)", "Margin of Safety (%)",
    "ROE (%)", "D/E", "P/E", "P/B",
]


def _highlight_verdict(row):
    color = "#1a7f37" if "Buy" in row["Verdict"] else "#cf222e" if "Sell" in row["Verdict"] else "#9a6700"
    return [f"color: {color}; font-weight: 600" if col == "Verdict" else "" for col in row.index]


st.dataframe(
    filtered[display_cols].reset_index(drop=True).style.apply(_highlight_verdict, axis=1),
    use_container_width=True,
    height=480,
)

st.markdown("### 🔍 বিস্তারিত স্কোর ব্রেকডাউন")
pick = st.selectbox("একটা স্টক বেছে নিন বিস্তারিত দেখতে", filtered["Ticker"] if not filtered.empty else df["Ticker"])
row = df[df["Ticker"] == pick].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Buffett Score", f"{row['Buffett Score']}/100")
col2.metric("Current Price", f"₹{row['Price (₹)']}")
mos_val = row["Margin of Safety (%)"]
col3.metric("Margin of Safety", f"{mos_val}%" if mos_val is not None else "N/A")
col4.metric("Cap Category", row["Cap Category"])
col5.markdown(f"**Verdict**  \n{row['Verdict']}")

with st.expander("এই Verdict কীভাবে হিসাব হলো?"):
    st.markdown(
        f"""
- 🟢 **Buy Zone**: Buffett Score ≥ {BUY_SCORE_MIN} এবং Margin of Safety > {BUY_MOS_MIN}% (মানে দাম Intrinsic Value থেকে অন্তত {BUY_MOS_MIN}% কম)
- 🔴 **Sell Zone**: Buffett Score < {SELL_SCORE_MAX} অথবা দাম Intrinsic Value থেকে {abs(SELL_MOS_MAX)}%+ বেশি (overvalued)
- 🟡 **Hold Zone**: বাকি সব ক্ষেত্রে — ফান্ডামেন্টাল মোটামুটি ভালো কিন্তু দাম fair value-এর কাছাকাছি
        """
    )

detail_df = pd.DataFrame(row["_rows"], columns=["Criterion", "Value", "Points", "Max"])
st.table(detail_df)

st.caption(
    f"সর্বশেষ ডেটা আপডেট (cache): {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    "⚠️ Buy/Hold/Sell ট্যাগটি সম্পূর্ণ একটি নিয়ম-ভিত্তিক (rule-based) হিসাব — শুধু Buffett Score ও "
    "Margin of Safety-এর উপর ভিত্তি করে তৈরি, ব্যক্তিগত বিনিয়োগ পরামর্শ নয়। কেনা-বেচার আগে নিজে গবেষণা "
    "করুন বা SEBI-নিবন্ধিত উপদেষ্টার পরামর্শ নিন।"
)
