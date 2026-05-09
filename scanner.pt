import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Logging Configuration (Same as v4) ────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration (Supports GitHub Secrets) ────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "your_email@gmail.com")

MIN_MARKET_CAP = 1_000_000_000
MAX_RSI        = 75             # Upgraded threshold
EMA_PERIOD     = 150
MAX_WORKERS    = 30

# ── Stock Universe (Expanded Sectors) ──────────────────────────────────────────
def get_stock_universe():
    try:
        # Core: S&P 500 (Dynamic)
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_tickers = sp500['Symbol'].tolist()
        
        # Sector: Semiconductors
        chips = ["NVDA", "AMD", "TSM", "AVGO", "ASML", "ARM", "MU", "INTC", "QCOM", "LRCX", "AMAT", "ADI", "TXN"]
        
        # Sector: Software & AI
        software = ["MSFT", "ORCL", "ADBE", "CRM", "PLTR", "SNOW", "PANW", "CRWD", "DDOG", "NET", "NOW", "MSTR"]
        
        # Sector: Quantum & Advanced Energy
        quantum = ["IONQ", "RGTI", "QBTS", "OKLO", "QUBT", "ARQQ"]
        
        # Sector: Industrials (Your favorites)
        industrials = ["CAT", "HON", "GE", "ETN", "VRT", "TT", "EMR", "WM"]

        combined = list(set([t.replace('.', '-') for t in sp500_tickers] + chips + software + quantum + industrials))
        logger.info(f"Scanning universe: {len(combined)} tickers (S&P500 + Tech/Industrials)")
        return combined
    except Exception as e:
        logger.error(f"Universe fetch failed: {e}")
        return ["NVDA", "AMD", "MSFT", "CAT", "IONQ", "HON"]

# ── Engine A: Trend (Structure from v4) ────────────────────────────────────────
def engine_a_trend(df):
    score = 0
    last = df.iloc[-1]
    
    # Fundamental Filter: Above EMA 150
    if last['Close'] > last['EMA150']:
        score += 15
    else:
        return 0 # Fail immediately
        
    # Extra Trend Confirmation (EMA20 > EMA50)
    if last['EMA20'] > last['EMA50']:
        score += 15
        
    return score

# ── Engine B: Momentum (CCI Cross & RSI 75) ─────────────────────────────────────
def engine_b_momentum(df):
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # CCI Accuracy: Crossover of the 35 line
    if prev['CCI'] <= 35 and last['CCI'] > 35:
        score += 25
    elif last['CCI'] > 35:
        score += 10
        
    # RSI Threshold: Up to 75
    if last['RSI'] <= MAX_RSI:
        score += 15
        
    return score

# ── Engine C: Patterns (Structure from v4) ─────────────────────────────────────
def engine_c_patterns(df):
    score = 0
    close = df['Close'].values
    
    # 1. Cup & Handle / Breakout Accuracy
    high_52w = np.max(close[-250:])
    if 0.95 <= (close[-1] / high_52w) <= 1.05:
        score += 15
        
    # 2. Inverse Head & Shoulders Detection
    lows = df['Low'].iloc[-60:].values
    head_idx = np.argmin(lows)
    if 15 < head_idx < 45: 
        score += 15
        
    return score

# ── Analysis Pipeline ─────────────────────────────────────────────────────────
def analyze_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if len(df) < EMA_PERIOD: return None
        
        info = stock.info
        if info.get('marketCap', 0) < MIN_MARKET_CAP: return None

        # Technical Indicators
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA150'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - ma) / (0.015 * mad)

        # Execute Parallel Engines
        s_a = engine_a_trend(df)
        if s_a == 0: return None
        
        s_b = engine_b_momentum(df)
        s_c = engine_c_patterns(df)
        
        total_score = s_a + s_b + s_c
        
        if total_score >= 55:
            return {
                "Ticker": ticker,
                "Price": round(df.iloc[-1]['Close'], 2),
                "Score": total_score,
                "RSI": round(df.iloc[-1]['RSI'], 1),
                "CCI": round(df.iloc[-1]['CCI'], 1),
                "Sector": info.get('sector', 'N/A'),
                "TradingView": f"https://www.tradingview.com/chart/?symbol={ticker}"
            }
    except:
        return None

# ── Email Styling (Preserving Original Design) ──────────────────────────────────
def send_email(df, scan_date
