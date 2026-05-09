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

# ── הגדרות לוגינג ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── קונפיגורציה (GitHub Secrets) ───────────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "your_email@gmail.com")

MIN_MARKET_CAP = 1_000_000_000 
MAX_RSI        = 75             # הרף המעודכן ל-75
EMA_PERIOD     = 150
MAX_WORKERS    = 30

# ── שדרוג יקום המניות (סקטורים מורחבים) ──────────────────────────────────────
def get_stock_universe():
    try:
        # 1. משיכת S&P 500 (מניות הליבה)
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500['Symbol'].tolist()
        
        # 2. סקטור השבבים (Semiconductors)
        chips = ["NVDA", "AMD", "TSM", "AVGO", "ASML", "ARM", "MU", "INTC", "QCOM", "LRCX", "AMAT", "ADI", "TXN"]
        
        # 3. סקטור התוכנה (Software & AI)
        software = ["MSFT", "ORCL", "ADBE", "CRM", "PLTR", "SNOW", "PANW", "CRWD", "DDOG", "NET", "NOW", "MSTR"]
        
        # 4. סקטור הקוונטום ואנרגיה מתקדמת (Quantum & Specialized Tech)
        quantum = ["IONQ", "RGTI", "QBTS", "OKLO", "QUBT", "ARQQ"]
        
        # 5. סקטור התעשייה (Industrials - המועדפים עליך)
        industrials = ["CAT", "HON", "GE", "ETN", "VRT", "TT", "EMR", "WM"]

        # איחוד כל הרשימות ומניעת כפילויות
        combined = list(set([t.replace('.', '-') for t in tickers] + chips + software + quantum + industrials))
        
        logger.info(f"Scanning {len(combined)} stocks across Chips, Software, Quantum and Industrials.")
        return combined
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return ["NVDA", "AMD", "MSFT", "CAT", "IONQ", "HON"] # Fallback

# ── שלושת מנועי הניתוח (לפי המבנה המקורי שלך) ───────────────────────────────────

def engine_a_trend(df):
    """מנוע א': פילטר מגמה - EMA 150"""
    score = 0
    last = df.iloc[-1]
    
    # תנאי סף: מחיר מעל ממוצע 150
    if last['Close'] > last['EMA150']:
        score += 20
        # בונוס על מגמה חזקה (EMA20 מעל EMA50)
        if last['EMA20'] > last['EMA50']:
            score += 10
    else:
        return 0 # פסילה מיידית אם מתחת ל-EMA150
    
    return score

def engine_b_momentum(df):
    """מנוע ב': מומנטום - CCI ו-RSI"""
    score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # חציית CCI את ה-35 כלפי מעלה (דיוק פריצה)
    if prev['CCI'] <= 35 and last['CCI'] > 35:
        score += 25
    elif last['CCI'] > 35:
        score += 10
        
    # RSI בטווח הבריא (עד 75)
    if last['RSI'] <= MAX_RSI:
        score += 15
        if last['RSI'] > 50: # בונוס על מומנטום חיובי אך לא מתוח
            score += 5
            
    return score

def engine_c_patterns(df):
    """מנוע ג': תבניות טכניות ו-VCP"""
    score = 0
    close = df['Close'].values
    
    # 1. זיהוי ספל וידית / פריצת שיא (VCP)
    high_52w = np.max(close[-250:])
    if 0.94 <= (close[-1] / high_52w) <= 1.03:
        score += 15
        
    # 2. זיהוי ראש וכתפיים הפוך (Inverse H&S)
    lows = df['Low'].iloc[-60:].values
    head_idx = np.argmin(lows)
    if 15 < head_idx < 45: 
        score += 15
        
    return score

# ── ניתוח מניה בודדת ─────────────────────────────────────────────────────────
def analyze_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if len(df) < EMA_PERIOD: return None
        
        info = stock.info
        if info.get('marketCap', 0) < MIN_MARKET_CAP: return None

        # חישוב אינדיקטורים
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA150'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - ma) / (0.015 * mad)

        # הרצת המנועים
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
                "Industry": info.get('industry', 'N/A'),
                "TV_Link": f"https://www.tradingview.com/chart/?symbol={ticker}"
            }
    except:
        return None

# ── שליחת דוח HTML (לפי המבנה המקורי שלך) ──────────────────────────────────────
def send_email(df, scan_date):
    if df.empty: return
    
    html_table = df.to_html(index=False, classes='table', render_links=True)
    html_body = f"""
    <div dir="rtl" style="font-family: Segoe UI, Tahoma, sans-serif;">
        <h2 style="color: #2c3e50;">דוח סריקה יומי - {scan_date}</h2>
        <p>נסרקו מניות מסקטור השבבים, תוכנה, קוונטום ותעשייה.</p>
        <p><b>קריטריונים:</b> מחיר > EMA150, RSI < 75, CCI חצה 35.</p>
        {html_table}
    </div>
    """
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Stock Scanner Report: {len(df)} Candidates Found ({scan_date})"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Email failed: {e}")

# ── הרצה ראשית ──────────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    scan_date = start_time.strftime("%d/%m/%Y")
    
    tickers = get_stock_universe()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)

    final_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    
    if not final_df.empty:
        # שמירת יומן מסחר מקומי
        final_df.to_csv(f"journal_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        send_email(final_df, scan_date)
        print(final_df.head(20))
    else:
        logger.warning("No matches found today.")

if __name__ == "__main__":
    main()
