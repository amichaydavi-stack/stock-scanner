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

# ── קונפיגורציה (תומך ב-GitHub Secrets) ──────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "your_email@gmail.com")

MIN_MARKET_CAP = 1_000_000_000 # מינימום מיליארד דולר
EMA_LONG       = 150
MAX_WORKERS    = 30

# ── משיכת רשימת מניות דינמית ────────────────────────────────────────────────
def get_dynamic_universe():
    try:
        # משיכת S&P 500 מוויקיפדיה
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500['Symbol'].tolist()
        
        # הוספת מניות ספציפיות מהמעקב שלך (תעשייה, טכנולוגיה וצמיחה)
        watch_list = ["CAT", "HON", "IONQ", "OKLO", "NVDA", "VRT", "ETN"]
        full_list = list(set([t.replace('.', '-') for t in tickers] + watch_list))
        
        logger.info(f"Scanning {len(full_list)} tickers (S&P 500 + Watchlist)")
        return full_list
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return ["AAPL", "MSFT", "NVDA", "CAT", "HON", "AMZN"]

# ── חישוב אינדיקטורים ודיוקים טכניים ──────────────────────────────────────────
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if len(df) < EMA_LONG: return None
        
        info = stock.info
        mkt_cap = info.get('marketCap', 0)
        if mkt_cap < MIN_MARKET_CAP: return None

        # חישוב EMA 150
        df['EMA150'] = df['Close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        # חישוב RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # חישוב CCI (20)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - ma) / (0.015 * mad)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        reasons = []

        # בדיקת תנאי סף: מחיר מעל EMA 150
        if last['Close'] < last['EMA150']: return None

        # דיוק CCI: חצייה כלפי מעלה של קו ה-35
        if prev['CCI'] <= 35 and last['CCI'] > 35:
            score += 40
            reasons.append("CCI Bullish Cross (35)")
        
        # דיוק RSI: מתחת ל-65 (לא קניית יתר)
        if last['RSI'] < 65:
            score += 20
            reasons.append("RSI Health (<65)")
        
        # זיהוי תבניות (מנוע משופר)
        # 1. ספל וידית - מחיר קרוב לשיא שנתי אך בגיבוש
        year_high = df['High'].rolling(window=250).max().iloc[-1]
        if 0.94 <= (last['Close'] / year_high) <= 1.02:
            score += 20
            reasons.append("Cup & Handle/High Tight Flag")

        # 2. ראש וכתפיים הפוך (Inverse H&S) - זיהוי שפלים
        lows = df['Low'].iloc[-60:].values
        if len(lows) > 40:
            head_idx = np.argmin(lows)
            if 10 < head_idx < 50: # הראש באמצע
                score += 20
                reasons.append("Inverse H&S Pattern")

        if score >= 60:
            return {
                "Ticker": ticker,
                "Price": round(last['Close'], 2),
                "Score": score,
                "RSI": round(last['RSI'], 1),
                "CCI": round(last['CCI'], 1),
                "Reasons": " | ".join(reasons),
                "Sector": info.get('sector', 'N/A'),
                "MarketCap_B": round(mkt_cap / 1e9, 2),
                "TV_Link": f"https://www.tradingview.com/chart/?symbol={ticker}"
            }
    except:
        return None

# ── שליחת דוח HTML (לפי המבנה שאתה אוהב) ──────────────────────────────────────
def send_email(df, scan_date):
    if df.empty: return
    
    html_table = df.to_html(index=False, classes='table', render_links=True)
    
    # עיצוב ה-HTML
    html_body = f"""
    <html>
    <head>
        <style>
            .table {{ font-family: Arial, sans-serif; border-collapse: collapse; width: 100%; }}
            .table td, .table th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table tr:nth-child(even){{background-color: #f2f2f2;}}
            .table th {{ background-color: #04AA6D; color: white; }}
            .high-score {{ color: #04AA6D; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>דוח סורק מניות יומי - {scan_date}</h2>
        <p>נמצאו {len(df)} מניות שעמדו בקריטריונים (EMA150, CCI Cross 35, RSI < 65):</p>
        {html_table}
        <br>
        <p><i>הדוח הופק אוטומטית על ידי Stock Scanner v5.</i></p>
    </body>
    </html>
    """
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Stock Scan: {len(df)} Opportunities Found ({scan_date})"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        logger.info("Email sent successfully!")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

# ── הרצה ראשית ──────────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    scan_date = start_time.strftime("%d/%m/%Y")
    logger.info(f"Starting Scan for {scan_date}")

    tickers = get_dynamic_universe()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, t): t for t in tickers}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    final_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    
    if not final_df.empty:
        # שמירה ליומן מסחר (CSV)
        final_df.to_csv(f"journal_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        send_email(final_df, scan_date)
    else:
        logger.info("No matches found today.")

if __name__ == "__main__":
    main()
