"""
Stock Scanner - Daily Momentum Scanner
Filters NASDAQ + NYSE stocks by market cap, EMA150, RSI < 70
Then scores by momentum: RSI, volume, moving averages
Sends daily email report
"""

import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "your_email@gmail.com")

MIN_MARKET_CAP   = 1_000_000_000   # $1B minimum
MAX_RSI          = 70
EMA_PERIOD       = 150
MAX_WORKERS      = 20               # parallel threads
TOP_N            = 20               # stocks in final report

# ── Stock Universe ─────────────────────────────────────────────────────────────

def get_stock_universe():
    tickers = [
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","JPM","V",
        "MA","UNH","XOM","LLY","JNJ","PG","HD","MRK","ABBV","COST","ORCL",
        "BAC","KO","PEP","ADBE","CRM","AMD","NFLX","TMO","ACN","MCD","CSCO",
        "ABT","WMT","DHR","NEE","TXN","PM","CAT","QCOM","AMGN","GE","INTU",
        "MS","IBM","RTX","SPGI","LOW","GS","ELV","ISRG","HON","AMAT","NOW",
        "BKNG","TJX","SYK","VRTX","ADP","MDLZ","GILD","MMC","BLK","PLD",
        "REGN","ADI","PANW","LRCX","MU","KLAC","SNPS","CDNS","MRVL","FICO",
        "FTNT","CRWD","WDAY","DDOG","ZS","NET","MDB","SNOW","PLTR","APP",
        "AXON","DECK","CELH","GDDY","TTD","HUBS","PAYC","VEEV","CPRT","MPWR",
        "WST","IDXX","PODD","ALGN","EW","DXCM","MTD","WAT","A","BIO",
        "JPM","BAC","WFC","C","GS","MS","AXP","BX","KKR","APO",
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
        "LIN","APD","SHW","ECL","DD","DOW","PPG","NUE","FCX","ALB",
        "DIS","CMCSA","CHTR","WBD","PARA","FOX","NYT","SPOT","RBLX","EA",
        "UNP","UPS","FDX","CSX","NSC","DAL","UAL","LUV","AAL","JBLU",
        "PFE","BMY","MRNA","BIIB","ILMN","IQV","CRL","CTLT","VTRS","ZBH",
        "AMT","PLD","CCI","EQIX","SPG","O","WELL","DLR","PSA","EXR"
    ]
    logger.info(f"Total unique tickers in universe: {len(tickers)}")
    return tickers

# ── Technical Indicators ───────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Single Stock Analysis ──────────────────────────────────────────────────────
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # ── Stage 1 Filter: Market Cap ─────────────────────────────────────────
        market_cap = info.get('marketCap', 0) or 0
        if market_cap < MIN_MARKET_CAP:
            return None

        # ── Download Price History ─────────────────────────────────────────────
        hist = stock.history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < EMA_PERIOD + 10:
            return None

        close  = hist['Close']
        high   = hist['High']
        low    = hist['Low']
        volume = hist['Volume']

        current_price = close.iloc[-1]
        if current_price <= 0:
            return None

        # ── Stage 2 Filter: Price > EMA150 ────────────────────────────────────
        ema150 = compute_ema(close, EMA_PERIOD).iloc[-1]
        if current_price <= ema150:
            return None

        # ── Stage 3 Filter: RSI < 70 ──────────────────────────────────────────
        rsi_series = compute_rsi(close, 14)
        rsi = rsi_series.iloc[-1]
        if rsi >= MAX_RSI or np.isnan(rsi):
            return None

        # ── Momentum Scoring ──────────────────────────────────────────────────
        ema20  = compute_ema(close, 20).iloc[-1]
        ema50  = compute_ema(close, 50).iloc[-1]
        ema200 = compute_ema(close, 200)

        # Volume surge (current vs 20-day avg)
        avg_volume_20  = volume.iloc[-21:-1].mean()
        current_volume = volume.iloc[-1]
        volume_ratio   = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        # Price momentum
        ret_1w  = (current_price / close.iloc[-6]  - 1) * 100 if len(close) >= 6  else 0
        ret_1m  = (current_price / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0
        ret_3m  = (current_price / close.iloc[-65] - 1) * 100 if len(close) >= 65 else 0

        # Distance from EMA150 (%)
        ema150_pct = (current_price / ema150 - 1) * 100

        # Golden cross: EMA20 > EMA50
        golden_cross = 1 if ema20 > ema50 else 0

        # Above EMA200
        above_ema200 = 1 if len(ema200) > 0 and current_price > ema200.iloc[-1] else 0

        # 52-week high proximity
        high_52w    = high.iloc[-252:].max() if len(high) >= 252 else high.max()
        pct_from_52w_high = (current_price / high_52w - 1) * 100

        # ── Momentum Score (0–100) ────────────────────────────────────────────
        score = 0
        score += min(rsi, 69) / 69 * 20           # RSI: high but < 70 is bullish
        score += min(volume_ratio, 5) / 5 * 20     # Volume surge
        score += min(max(ret_1m, 0), 30) / 30 * 20 # 1-month return
        score += min(max(ret_3m, 0), 50) / 50 * 15 # 3-month return
        score += golden_cross * 10                  # Golden cross bonus
        score += above_ema200 * 10                  # Above EMA200
        score += min(max(ema150_pct, 0), 20) / 20 * 5  # Trend strength

        # ── Fundamentals ──────────────────────────────────────────────────────
        pe_ratio   = info.get('trailingPE')
        ps_ratio   = info.get('priceToSalesTrailing12Months')
        revenue_gr = info.get('revenueGrowth')
        sector     = info.get('sector', 'N/A')
        name       = info.get('shortName', ticker)

        return {
            'ticker':          ticker,
            'name':            name,
            'sector':          sector,
            'price':           round(current_price, 2),
            'market_cap_b':    round(market_cap / 1e9, 1),
            'rsi':             round(rsi, 1),
            'ema150':          round(ema150, 2),
            'ema150_pct':      round(ema150_pct, 1),
            'ema20_50_cross':  '✅' if golden_cross else '❌',
            'above_ema200':    '✅' if above_ema200 else '❌',
            'volume_ratio':    round(volume_ratio, 2),
            'ret_1w_pct':      round(ret_1w, 1),
            'ret_1m_pct':      round(ret_1m, 1),
            'ret_3m_pct':      round(ret_3m, 1),
            'pct_from_52w_high': round(pct_from_52w_high, 1),
            'pe_ratio':        round(pe_ratio, 1) if pe_ratio else 'N/A',
            'ps_ratio':        round(ps_ratio, 1) if ps_ratio else 'N/A',
            'revenue_growth':  f"{round(revenue_gr*100,1)}%" if revenue_gr else 'N/A',
            'momentum_score':  round(score, 1),
        }

    except Exception as e:
        logger.debug(f"Error analyzing {ticker}: {e}")
        return None


# ── Scan All Stocks ────────────────────────────────────────────────────────────
def run_scan(tickers):
    results = []
    logger.info(f"Starting scan of {len(tickers)} tickers with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(tickers)} scanned, {len(results)} passed filters")

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values('momentum_score', ascending=False).reset_index(drop=True)
    return df


# ── Email Report ───────────────────────────────────────────────────────────────
def build_html_report(df, scan_date):
    top = df.head(TOP_N)
    
    rows_html = ""
    for i, row in top.iterrows():
        rank = i + 1
        score_color = "#22c55e" if row['momentum_score'] >= 70 else "#f59e0b" if row['momentum_score'] >= 50 else "#94a3b8"
        rows_html += f"""
        <tr>
          <td style="text-align:center;font-weight:700;color:#94a3b8;">{rank}</td>
          <td>
            <strong style="color:#f1f5f9;font-size:15px;">{row['ticker']}</strong><br>
            <span style="color:#94a3b8;font-size:12px;">{row['name']}</span>
          </td>
          <td style="color:#94a3b8;font-size:12px;">{row['sector']}</td>
          <td style="text-align:right;color:#f1f5f9;">${row['price']}</td>
          <td style="text-align:right;color:#f1f5f9;">${row['market_cap_b']}B</td>
          <td style="text-align:center;color:#f59e0b;font-weight:700;">{row['rsi']}</td>
          <td style="text-align:center;color:#22c55e;">{row['ema150_pct']}%</td>
          <td style="text-align:center;">{row['ema20_50_cross']}</td>
          <td style="text-align:center;">{row['above_ema200']}</td>
          <td style="text-align:center;color:#60a5fa;">{row['volume_ratio']}x</td>
          <td style="text-align:right;color:{'#22c55e' if row['ret_1w_pct'] > 0 else '#f87171'};">{row['ret_1w_pct']}%</td>
          <td style="text-align:right;color:{'#22c55e' if row['ret_1m_pct'] > 0 else '#f87171'};">{row['ret_1m_pct']}%</td>
          <td style="text-align:right;color:{'#22c55e' if row['ret_3m_pct'] > 0 else '#f87171'};">{row['ret_3m_pct']}%</td>
          <td style="text-align:center;color:#94a3b8;">{row['pe_ratio']}</td>
          <td style="text-align:center;">
            <span style="background:{score_color};color:#0f172a;font-weight:800;padding:3px 10px;border-radius:20px;font-size:13px;">{row['momentum_score']}</span>
          </td>
        </tr>"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ margin:0; padding:0; background:#0f172a; font-family: 'Helvetica Neue', Arial, sans-serif; color:#f1f5f9; }}
  .container {{ max-width:1100px; margin:0 auto; padding:30px 20px; }}
  h1 {{ color:#f1f5f9; font-size:28px; margin:0 0 4px; letter-spacing:-0.5px; }}
  .subtitle {{ color:#64748b; font-size:14px; margin-bottom:30px; }}
  .badge {{ display:inline-block; background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px 18px; margin:0 8px 16px 0; font-size:13px; color:#94a3b8; }}
  .badge strong {{ color:#f1f5f9; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#1e293b; color:#64748b; padding:10px 8px; text-align:left; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid #334155; }}
  td {{ padding:10px 8px; border-bottom:1px solid #1e293b; }}
  tr:hover td {{ background:#1e293b44; }}
  .footer {{ margin-top:30px; color:#475569; font-size:12px; text-align:center; }}
</style>
</head>
<body>
<div class="container">
  <h1>📈 Daily Stock Momentum Scanner</h1>
  <p class="subtitle">{scan_date} — NASDAQ & NYSE</p>
  
  <div>
    <div class="badge">סך הכל עברו סינון: <strong>{len(df)}</strong> מניות</div>
    <div class="badge">מוצגות Top: <strong>{min(TOP_N, len(df))}</strong></div>
    <div class="badge">פילטרים: <strong>Market Cap &gt;$1B | מחיר &gt; EMA150 | RSI &lt; 70</strong></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>מניה</th>
        <th>סקטור</th>
        <th>מחיר</th>
        <th>שווי שוק</th>
        <th>RSI</th>
        <th>מעל EMA150</th>
        <th>EMA20&gt;50</th>
        <th>EMA200</th>
        <th>נפח</th>
        <th>1W</th>
        <th>1M</th>
        <th>3M</th>
        <th>P/E</th>
        <th>ציון</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="footer">
    <p>⚠️ הדוח הזה הוא לצורכי מידע בלבד ואינו מהווה המלצת השקעה.<br>
    נוצר אוטומטית על ידי Stock Momentum Scanner • {scan_date}</p>
  </div>
</div>
</body>
</html>"""
    return html


def send_email(html_body, scan_date, total_scanned, passed_filter):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"📈 סורק מניות יומי — {scan_date} | {passed_filter} מניות עברו סינון"
    msg['From']    = EMAIL_SENDER
    msg['To']      = EMAIL_RECEIVER

    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        logger.info(f"Email sent successfully to {EMAIL_RECEIVER}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    scan_date = datetime.now().strftime("%d/%m/%Y %H:%M")
    logger.info(f"=== Stock Scanner starting: {scan_date} ===")

    tickers = get_stock_universe()
    df      = run_scan(tickers)

    if df.empty:
        logger.warning("No stocks passed the filters today.")
        # Still send an email to notify
        html = f"<h2>סורק מניות — {scan_date}</h2><p>לא נמצאו מניות שעברו את הפילטרים היום.</p>"
        send_email(html, scan_date, len(tickers), 0)
        return

    logger.info(f"Scan complete: {len(df)} stocks passed all filters")

    # Save CSV
    output_file = f"scan_results_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    # Send email
    html = build_html_report(df, scan_date)
    send_email(html, scan_date, len(tickers), len(df))

    logger.info("=== Scanner finished successfully ===")


if __name__ == "__main__":
    main()
