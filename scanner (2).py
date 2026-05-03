"""
Stock Scanner v4 — Parallel Pipeline Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1 — Fast pre-filter (market cap, volume, price)
Stage 2 — 3 parallel engines:
    Engine A: Trend    (EMA150, EMA200, EMA20>50)        max 30 pts
    Engine B: Momentum (RSI, MACD, Stochastic, Volume)   max 40 pts
    Engine C: Patterns (8 chart patterns + volume valid) max 30 pts
Stage 3 — Combine scores, keep only stocks > 60 pts
Stage 4 — Send HTML email with TradingView links

Improvements over v3:
    • ~150 Small/Mid Cap stocks added (Russell 2000 leaders)
    • Pattern engine: volume confirmation required for every pattern
    • Pattern engine: real breakout check (price closed above, not just touched)
    • New patterns: Bullish Flag, Double Bottom, VCP
"""

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "your_email@gmail.com")

MIN_MARKET_CAP    = 1_000_000_000   # $1B
MIN_AVG_VOLUME    = 500_000         # 500K shares/day minimum
MIN_PRICE         = 5.0             # No penny stocks
MIN_FINAL_SCORE   = 60              # Only stocks passing all 3 engines
MAX_WORKERS       = 25
TOP_N             = 25

# Volume confirmation thresholds for patterns
BREAKOUT_VOL_MULT = 1.4   # Breakout bar must have 1.4x average volume
PATTERN_VOL_MULT  = 1.2   # General pattern volume confirmation


# ── Stock Universe ─────────────────────────────────────────────────────────────
def get_stock_universe():
    tickers = [
        # ── Mega Cap Tech & Semiconductors ────────────────────────────────────
        "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ORCL",
        "ADBE","CRM","CSCO","QCOM","TXN","AMD","MU","AMAT","LRCX","KLAC",
        "SNPS","CDNS","MRVL","MPWR","ANSS","ENTG","ONTO","COHR","WOLF","SWKS",
        "QRVO","ACLS","MKSI","IPGP","IIVI","FORM","ICHR","UCTT","CAMT","PLAB",
        "INTC","IBM","HPE","STM","NXPI","ON","MCHP","TEL","APH","NBIS","SAP",

        # ── Cloud / SaaS ──────────────────────────────────────────────────────
        "NOW","PANW","CRWD","FTNT","ZS","NET","OKTA","S","QLYS","TENB",
        "WDAY","INTU","ADSK","DDOG","SNOW","MDB","GTLB","CFLT","ESTC","NEWR",
        "DT","FIVN","NICE","TOST","BILL","HUBS","TTD","APP","RBLX","U",
        "PLTR","PATH","ASAN","DOCN","NCNO","ALTR","BRZE","CWAN","KVYO","RDDT",

        # ── Momentum / High RS / Market Leaders 2026 ─────────────────────────
        "AXON","DECK","CELH","GDDY","PAYC","VEEV","CPRT","FICO","IDXX","PODD",
        "ALGN","EW","DXCM","MTD","WAT","WST","BIO","NTRA","EXAS","PCVX",
        "IRTC","INSP","TMDX","ATEC","NVCR","NOVT","OMCL","ITGR","UFPT","LBRT",
        "CIVI","CHRD","PDCE","MTDR","CTRA","SM","VTLE","KOS","RRC",
        "SNDK","TPL","MRNA","GLW","TER","WDC","STX","CIEN","FIX","VRT","SMCI","DELL","HBM",

        # ── AI Power, Utilities & Infrastructure ─────────────────────────────
        "GEV","VST","CEG","NRG","NEE","DUK","SO","AEP","SRE","D","FE","PCG","ETR",

        # ── Quantum Computing ─────────────────────────────────────────────────
        # Pure-play quantum hardware & software
        "IONQ",   # IonQ — trapped-ion quantum computers
        "RGTI",   # Rigetti Computing — superconducting qubits
        "QUBT",   # Quantum Computing Inc — quantum software
        "ARQQ",   # Arqit Quantum — quantum encryption
        "QBTS",   # D-Wave Quantum — annealing quantum systems
        "QTUM",   # Defiance Quantum ETF (שימושי לסנטימנט סקטור)
        # קוונטום כחלק מפעילות רחבה יותר
        "IBM",    # IBM Quantum — מובילה במחשוב קוונטי (כבר ביקום)
        "GOOGL",  # Google Sycamore (כבר ביקום)
        "MSFT",   # Azure Quantum (כבר ביקום)
        "HONF",   # Honeywell Quantum (Quantinuum — לא ציבורית עדיין)
        "NVDA",   # NVIDIA — סימולציית קוונטום על GPU (כבר ביקום)
        "MKSI",   # MKS Instruments — ציוד למעבדות קוונטום
        "COHU",   # Cohu — בדיקות שבבים לקוונטום
        "VIAV",   # Viavi Solutions — פוטוניקה לקוונטום
        "PSIX",   # Power Solutions — אספקת חשמל מדויקת למערכות קוונטום
        "QLGN",   # Qualigen — quantum dots (ביו-קוונטום)

        # ── Nuclear Energy & SMR (Small Modular Reactors) ─────────────────────
        # Developers & operators
        "OKLO",   # Aurora SMR — backed by Sam Altman
        "NNE",    # Nano Nuclear Energy — micro reactors
        "SMR",    # NuScale Power — first NRC-approved SMR
        "BWXT",   # BWX Technologies — nuclear components & fuel
        "LEU",    # Centrus Energy — uranium enrichment
        "CCJ",    # Cameco — largest uranium miner
        "UEC",    # Uranium Energy Corp
        "UUUU",   # Energy Fuels — uranium + rare earths
        "DNN",    # Denison Mines — uranium
        "URG",    # Ur-Energy — uranium miner
        "EU",     # enCore Energy — uranium ISR
        "LTBR",   # Lightbridge — nuclear fuel technology
        "NFGC",   # Nano Nuclear (alt ticker check)
        "GEV",    # GE Vernova — nuclear + power infra (already above, dedup handles it)
        "ETN",    # Eaton — power mgmt for nuclear/data centers
        "PWR",    # Quanta Services — nuclear plant construction
        "AMRC",   # Ameresco — clean energy infra
        "X",      # US Steel — steel for reactor vessels (SMR supply chain)
        "CW",     # Curtiss-Wright — nuclear components
        "TDY",    # Teledyne — nuclear instrumentation

        # ── Financials, Payments & Insurance ─────────────────────────────────
        "JPM","BAC","WFC","C","GS","MS","AXP","BX","KKR","APO",
        "ARES","CG","TPG","HLNE","STEP","HOOD","COIN","SOFI","AFRM","NU",
        "PYPL","SQ","V","MA","SPGI","MCO","ICE","CME","CBOE","NDAQ",
        "FDS","MSCI","VRSK","BR","NTRS","STT","BK","TRV","CB","AON",
        "PGR","ALL","MMC","AJG","WRB","ACGL","AMP","TROW","BEN","IVZ",

        # ── Healthcare & Biotech ──────────────────────────────────────────────
        "LLY","UNH","JNJ","MRK","ABBV","TMO","DHR","ABT","ISRG","SYK",
        "ELV","VRTX","REGN","AMGN","GILD","BIIB","MRNA","BNTX","ALNY","SRPT",
        "RARE","ACAD","IONS","FOLD","KYMR","IMVT","CORT","PRAX","NRIX","RVMD",
        "EXEL","MRUS","ARWR","RCUS","PMVP","IGMS","KROS","ACHR","BEAM","EDIT",

        # ── Consumer, Retail & Leisure ────────────────────────────────────────
        "HD","MCD","SBUX","NKE","LULU","BKNG","ABNB","UBER","DASH","DUOL",
        "CHWY","ETSY","CVNA","AN","KMX","CACC","ALLY","TJX","COST","WMT",
        "TGT","ULTA","ELF","ONON","CROX","SKX","DECK","WWW","COLM","PVH",
        "RL","TPR","CPRI","VFC","HBI","GIL","LEVI","URBN","ANF","AEO",

        # ── Industrials, Transport & Logistics ───────────────────────────────
        "CAT","HON","GE","RTX","LMT","NOC","TDG","HWM","PWR","STRL",
        "ROAD","MTZ","EME","FIX","ESAB","TT","CARR","OTIS","AME","PH",
        "ROK","SWK","SNA","GGG","GNRC","FELE","AAON","MIDD","LYTS",
        "FDX","UPS","WM","RSG","CMI","UNP","CSX","NSC","CP","CNI","DE","PCAR",
        "GWW","FAST","URI","VMC","MLM","TYL","CSGP",

        # ── Energy ────────────────────────────────────────────────────────────
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
        "DVN","FANG","MRO","APA","HAL","BKR","NOV","SM","MTDR","CIVI",

        # ── Materials ─────────────────────────────────────────────────────────
        "LIN","APD","SHW","ECL","FCX","NUE","ALB","MP","USLM","VMC",
        "MLM","CRH","EXP","SUM","USCR","IIIN","CMC","RS","ATI","HCC",

        # ── REITs ─────────────────────────────────────────────────────────────
        "AMT","PLD","CCI","EQIX","SPG","O","WELL","DLR","PSA","EXR",
        "CUBE","LSI","NSA","REXR","TRNO","LXP","EGP","FR","STAG","COLD",

        # ══════════════════════════════════════════════════════════════════════
        # NEW — Small/Mid Cap Leaders (Russell 2000 High RS)
        # ══════════════════════════════════════════════════════════════════════

        # Small Cap Tech & Software
        "RIOT","MARA","CLSK","IREN","CIFR",           # Bitcoin miners
        "SMAR","APPF","PCTY","MANH","JAMF",            # SaaS mid-cap
        "ALKT","ALCO","CDAY","INFA","XPEL",
        "TASK","RAMP","FLYW","RELY","ZETA",
        "IONQ","RGTI","QUBT","ARQQ",                   # Quantum computing
        "AIOT","KTOS","BBAI","GFAI","AMBA",            # AI/Defense tech

        # Small Cap Semiconductors
        "ALGM","OLED","SITM","AXTI","DAKT",
        "DIOD","IMOS","SMTC","AOSL","MTSI",
        "KLIC","COHU","AEIS","POWI","ARLO",

        # Small Cap Healthcare / MedTech
        "PRCT","SWAV","NVCR","IRTC","INSP",
        "AXNX","NARI","IART","MMSI","LNTH",
        "ICAD","HALO","ITCI","ACVA","HRMY",
        "SNDX","IMCR","TMDX","RXRX","DNLI",
        "NUVL","VERA","ARRY","KRYS","ARDX",

        # Small Cap Financials
        "COOP","PFSI","GHLD","UWMC","RKT",
        "HBAN","CFG","WTFC","IBOC","CADE",
        "VRTS","GCMG","HLNE","MFIN","IIPR",

        # Small Cap Industrials / Defense
        "KTOS","RKLB","LUNR","RDW","SPCE",
        "ACHR","JOBY","LILM","EVTL","WKHS",
        "ATRO","MOOG","HEI","TDY","CW",
        "ESAB","NVST","GTLS","HLIO","TRS",

        # Small Cap Consumer / Restaurants
        "CAVA","BROS","SHAK","TXRH","WINGSTOP",
        "WING","FAT","DENN","JACK","LQDT",
        "YELP","TRIP","EXPE","MMYT","DESP",
        "PRPL","LOVE","LESL","XPOF","PLNT",

        # Small Cap Energy / Clean Tech
        "DUNE","ARRY","FTCI","SHLS","NOVA",
        "RUN","SPWR","FSLR","ENPH","SEDG",
        "GNRC","BE","PLUG","BLDP","FCEL",
        "REX","CIVI","BATL","TPVG","HESM",

        # Small Cap Materials / Mining
        "PAAS","CDE","HL","EGO","GATO",
        "MAG","SILV","WPM","AEM","KGC",
        "BTG","NGD","USAS","SAND","OR",

        # Mid Cap Retail / E-commerce
        "CPNG","SE","MELI","GRAB","GLBE",
        "POWL","ACMR","FORM","CRUS","OSIS",
        "SMPL","HIMS","SKIN","ELF","OLPX",
        "DNOW","KFRC","WLFC","HTLD","MRTN",
    ]

    tickers = list(set([t for t in tickers if isinstance(t, str) and 1 <= len(t) <= 5]))
    logger.info(f"Universe size: {len(tickers)} tickers")
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

def compute_macd(series):
    ema12  = compute_ema(series, 12)
    ema26  = compute_ema(series, 26)
    macd   = ema12 - ema26
    signal = compute_ema(macd, 9)
    hist   = macd - signal
    return macd, signal, hist

def compute_stochastic(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k   = 100 * (close - lowest) / (highest - lowest + 1e-10)
    pct_d   = pct_k.rolling(d).mean()
    return pct_k, pct_d

def vol_confirmed(volume, lookback=20, multiplier=PATTERN_VOL_MULT):
    """Check if the last bar has above-average volume (pattern confirmation)."""
    avg_vol = volume.iloc[-lookback-1:-1].mean()
    return volume.iloc[-1] > avg_vol * multiplier, round(volume.iloc[-1] / avg_vol, 2) if avg_vol > 0 else 0

def is_real_breakout(close, high, resistance, vol_ratio_required=BREAKOUT_VOL_MULT, volume=None, lookback=20):
    """
    True breakout = close (not just high) above resistance,
    confirmed by above-average volume on the breakout bar.
    """
    price_broke = close.iloc[-1] > resistance
    if not price_broke:
        return False
    if volume is not None:
        avg_vol = volume.iloc[-lookback-1:-1].mean()
        vol_ok  = volume.iloc[-1] > avg_vol * vol_ratio_required
        return vol_ok
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE A — TREND (max 30 pts)
# ══════════════════════════════════════════════════════════════════════════════
def engine_trend(close, high, low, volume):
    score    = 0
    details  = {}
    price    = close.iloc[-1]

    ema20  = compute_ema(close, 20)
    ema50  = compute_ema(close, 50)
    ema150 = compute_ema(close, 150)
    ema200 = compute_ema(close, 200)

    # Price > EMA150 (required filter + score)
    ema150_val = ema150.iloc[-1]
    if price <= ema150_val:
        return None, {}
    ema150_pct = (price / ema150_val - 1) * 100
    score += min(ema150_pct / 20, 1) * 10

    # Price > EMA200
    ema200_val   = ema200.iloc[-1]
    above_ema200 = price > ema200_val
    if above_ema200:
        score += 8

    # EMA20 > EMA50 (Golden Cross)
    golden_cross = ema20.iloc[-1] > ema50.iloc[-1]
    if golden_cross:
        score += 7

    # EMA slope (trending up)
    ema50_slope = (ema50.iloc[-1] - ema50.iloc[-10]) / ema50.iloc[-10] * 100
    if ema50_slope > 0:
        score += min(ema50_slope / 2, 1) * 5

    details = {
        'ema150_pct':   round(ema150_pct, 1),
        'above_ema200': '✅' if above_ema200 else '❌',
        'golden_cross': '✅' if golden_cross else '❌',
        'ema50_slope':  round(ema50_slope, 2),
        'trend_score':  round(score, 1),
    }
    return round(score, 1), details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE B — MOMENTUM (max 40 pts)
# ══════════════════════════════════════════════════════════════════════════════
def engine_momentum(close, high, low, volume):
    score   = 0
    details = {}
    price   = close.iloc[-1]

    # RSI
    rsi = compute_rsi(close, 14).iloc[-1]
    if np.isnan(rsi) or rsi >= 70:
        return None, {}
    if rsi >= 40:
        score += (rsi - 40) / 30 * 15

    # MACD
    macd, signal, hist = compute_macd(close)
    macd_bull  = macd.iloc[-1] > signal.iloc[-1]
    macd_cross = macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]
    if macd_bull:
        score += 8
    if macd_cross:
        score += 5

    # Stochastic
    pct_k, pct_d = compute_stochastic(high, low, close)
    stoch_k    = pct_k.iloc[-1]
    stoch_bull = stoch_k > pct_d.iloc[-1] and stoch_k < 80
    if stoch_bull:
        score += 5

    # Volume surge
    avg_vol   = volume.iloc[-21:-1].mean()
    vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1
    if vol_ratio >= MIN_AVG_VOLUME / 500_000:
        score += min(vol_ratio / 5, 1) * 7

    # Price momentum
    ret_1w = (price / close.iloc[-6]  - 1) * 100 if len(close) >= 6  else 0
    ret_1m = (price / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0
    ret_3m = (price / close.iloc[-65] - 1) * 100 if len(close) >= 65 else 0

    score += min(max(ret_1m, 0) / 30, 1) * 3
    score += min(max(ret_3m, 0) / 50, 1) * 2

    details = {
        'rsi':        round(rsi, 1),
        'macd_bull':  '✅' if macd_bull else '❌',
        'macd_cross': '🔥 חצייה טרייה' if macd_cross else '',
        'stoch_k':    round(stoch_k, 1),
        'vol_ratio':  round(vol_ratio, 2),
        'ret_1w':     round(ret_1w, 1),
        'ret_1m':     round(ret_1m, 1),
        'ret_3m':     round(ret_3m, 1),
        'mom_score':  round(score, 1),
    }
    return round(score, 1), details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE C — PATTERNS v4 (max 30 pts)
#
# כל תבנית דורשת:
#   1. מבנה מחיר תקין
#   2. ולידציה עם נפח מסחר
#   3. בדיקת breakout אמיתי (close מעל ההתנגדות, לא רק געגוע)
# ══════════════════════════════════════════════════════════════════════════════
def engine_patterns(close, high, low, volume):
    score    = 0
    patterns = []
    n        = len(close)
    if n < 60:
        return 0, []

    avg_vol_20 = volume.iloc[-21:-1].mean()

    # ── 1. Cup & Handle (10 pts) ──────────────────────────────────────────────
    # מבנה: ירידה עגולה → שחזור → ידית קטנה → פריצה מעל שוליים שמאליים
    # ולידציה: נפח הידית נמוך מהספל; פריצה עם נפח גבוה
    try:
        w          = close.iloc[-60:].values
        wv         = volume.iloc[-60:].values
        left_high  = w[:20].max()
        cup_low    = w[20:45].min()
        right_high = w[45:].max()
        handle_min = w[45:].min()
        handle_drop = (right_high - handle_min) / right_high if right_high > 0 else 1
        depth       = (left_high - cup_low) / left_high if left_high > 0 else 1

        # נפח: הידית (20 יום אחרונים) חייב להיות נמוך יחסית לספל
        cup_vol_avg    = wv[20:45].mean()
        handle_vol_avg = wv[45:].mean()
        vol_contraction = handle_vol_avg < cup_vol_avg * 1.1

        # פריצה אמיתית: close מעל left_high עם נפח
        real_bo = is_real_breakout(close, high, left_high * 0.995, volume=volume)

        if (0.12 <= depth <= 0.50
                and right_high >= left_high * 0.95
                and handle_drop <= 0.15
                and vol_contraction
                and real_bo):
            patterns.append("☕ ספל וידית")
            score += 10
    except:
        pass

    # ── 2. Inverse Head & Shoulders (10 pts) ─────────────────────────────────
    # מבנה: שתי כתפיים שווות + ראש נמוך יותר + פריצת קו צוואר
    # ולידציה: נפח יורד לאורך התבנית, פריצה עם נפח חזק
    try:
        w    = close.iloc[-50:].values
        wv   = volume.iloc[-50:].values
        s1   = w[0:10].min()
        head = w[15:30].min()
        s2   = w[35:50].min()
        neckline = (w[0:10].max() + w[35:50].max()) / 2

        # נפח יורד לאורך התבנית (אופייני ל-IH&S)
        vol_s1   = wv[0:10].mean()
        vol_head = wv[15:30].mean()
        vol_trend_ok = vol_head <= vol_s1 * 1.1   # נפח ראש לא גבוה מכתף שמאל

        # פריצת צוואר אמיתית עם נפח
        real_bo = is_real_breakout(close, high, neckline * 0.98, volume=volume)

        if (head < s1 * 0.97
                and head < s2 * 0.97
                and abs(s1 - s2) / s1 < 0.05
                and vol_trend_ok
                and real_bo):
            patterns.append("🔄 ראש וכתפיים הפוך")
            score += 10
    except:
        pass

    # ── 3. Flat Base (6 pts) ──────────────────────────────────────────────────
    # מבנה: קונסולידציה צרה (<12%) + פריצה עם נפח
    try:
        w              = close.iloc[-30:].values
        wv             = volume.iloc[-30:].values
        base_high      = w.max()
        depth          = (base_high - w.min()) / base_high if base_high > 0 else 1

        avg_base_vol   = wv[:-5].mean()
        avg_recent_vol = wv[-5:].mean()

        # פריצה: close מעל גובה הבסיס עם נפח
        real_bo = is_real_breakout(close, high, base_high * 0.97, volume=volume)

        if (depth <= 0.12
                and avg_recent_vol > avg_base_vol * 1.3
                and real_bo):
            patterns.append("📏 בסיס שטוח")
            score += 6
    except:
        pass

    # ── 4. Ascending Triangle (6 pts) ────────────────────────────────────────
    # מבנה: התנגדות אופקית + תמיכה עולה + פריצה עם נפח
    try:
        w_high  = high.iloc[-40:].values
        w_close = close.iloc[-40:].values
        w_low   = low.iloc[-40:].values

        resistance = w_high[:20].max()

        # בדיקת higher lows (תמיכה עולה)
        lows = [w_close[i] for i in range(0, 40, 8)]
        rising = all(lows[i] < lows[i+1] for i in range(len(lows)-1))

        # נגיעות בהתנגדות (לפחות 2)
        touches = sum(1 for h in w_high if h >= resistance * 0.98)

        # פריצה אמיתית: close מעל ההתנגדות עם נפח
        real_bo = is_real_breakout(close, high, resistance * 0.99, volume=volume)

        if rising and touches >= 2 and real_bo:
            patterns.append("📐 משולש עולה")
            score += 6
    except:
        pass

    # ── 5. 52-Week Breakout (8 pts) ───────────────────────────────────────────
    # פריצת שיא שנה: close מעל השיא עם נפח לפחות 1.5x ממוצע
    try:
        high_52w = high.iloc[-252:].max() if len(high) >= 252 else high.max()

        # close מעל שיא (לא רק high) + נפח חזק
        real_bo = is_real_breakout(
            close, high, high_52w * 0.995,
            vol_ratio_required=1.5,
            volume=volume
        )

        if real_bo:
            patterns.append("🚀 פריצת שיא 52 שבועות")
            score += 8
    except:
        pass

    # ── 6. Bullish Flag (8 pts) — NEW ─────────────────────────────────────────
    # מבנה: עמוד (עלייה חדה) + דגל (קונסולידציה יורדת קלה) + פריצה
    # ולידציה: נפח גבוה בעמוד, נפח נמוך בדגל, נפח גבוה בפריצה
    try:
        # עמוד: 10 יום לפני הדגל — עלייה של לפחות 8%
        pole_start = close.iloc[-25]
        pole_end   = close.iloc[-15]
        pole_gain  = (pole_end / pole_start - 1) * 100 if pole_start > 0 else 0

        # דגל: 10-15 יום אחרונים — ירידה עדינה (<6%) בנפח נמוך
        flag_high  = close.iloc[-15:-1].max()
        flag_low   = close.iloc[-15:-1].min()
        flag_depth = (flag_high - flag_low) / flag_high if flag_high > 0 else 1

        # נפח: דגל שקט (מתחת לממוצע)
        pole_vol_avg  = volume.iloc[-25:-15].mean()
        flag_vol_avg  = volume.iloc[-15:-1].mean()
        flag_quiet    = flag_vol_avg < pole_vol_avg * 0.85

        # פריצה: close מעל גובה הדגל עם נפח
        real_bo = is_real_breakout(close, high, flag_high * 0.99, volume=volume)

        if (pole_gain >= 8
                and flag_depth <= 0.06
                and flag_quiet
                and real_bo):
            patterns.append("🚩 דגל שורי")
            score += 8
    except:
        pass

    # ── 7. Double Bottom (7 pts) — NEW ───────────────────────────────────────
    # מבנה: שני שפלים שווים + פריצת קו הצוואר עם נפח
    # ולידציה: השפל השני לא נמוך מהראשון ב>3%; פריצה עם נפח
    try:
        w  = close.iloc[-50:].values
        wv = volume.iloc[-50:].values

        # מצא את שני השפלים
        bottom1_idx = np.argmin(w[:25])
        bottom2_idx = 25 + np.argmin(w[25:])
        bottom1     = w[bottom1_idx]
        bottom2     = w[bottom2_idx]

        # צוואר = הגובה בין שני השפלים
        neckline_db = w[bottom1_idx:bottom2_idx].max() if bottom2_idx > bottom1_idx else w.max()

        # תנאים: שפלים שווים יחסית, צוואר לפחות 5% מעל השפלים
        bottoms_equal = abs(bottom1 - bottom2) / bottom1 < 0.04 if bottom1 > 0 else False
        neck_height   = (neckline_db - bottom1) / bottom1 * 100 if bottom1 > 0 else 0

        # נפח: השפל השני נמוך יותר בנפח מהראשון (אופייני)
        vol_b1 = wv[max(0, bottom1_idx-2):bottom1_idx+3].mean()
        vol_b2 = wv[max(0, bottom2_idx-2):bottom2_idx+3].mean()
        vol_ok = vol_b2 <= vol_b1 * 1.1

        # פריצת צוואר אמיתית
        real_bo = is_real_breakout(close, high, neckline_db * 0.99, volume=volume)

        if (bottoms_equal
                and neck_height >= 5
                and vol_ok
                and real_bo):
            patterns.append("⚡ תחתית כפולה")
            score += 7
    except:
        pass

    # ── 8. VCP — Volatility Contraction Pattern (9 pts) — NEW ────────────────
    # מבנה (Minervini): סדרת קונסולידציות מצטמצמות עם נפח יורד
    # כל שלב: עומק ירידה קטן יותר מהקודם, נפח יורד → פריצה עם נפח גבוה
    try:
        # 3 חלונות של 15 יום — בדוק שהעומק יורד בכל שלב
        w1 = close.iloc[-45:-30].values
        w2 = close.iloc[-30:-15].values
        w3 = close.iloc[-15:-1].values

        v1 = volume.iloc[-45:-30].mean()
        v2 = volume.iloc[-30:-15].mean()
        v3 = volume.iloc[-15:-1].mean()

        def depth_pct(arr):
            mx = arr.max()
            return (mx - arr.min()) / mx * 100 if mx > 0 else 100

        d1, d2, d3 = depth_pct(w1), depth_pct(w2), depth_pct(w3)

        # כל שלב צר יותר מהקודם (לפחות 20% צמצום בכל פעם)
        contracting_price = d1 > d2 * 1.20 and d2 > d3 * 1.20

        # נפח יורד בכל שלב
        contracting_vol = v1 > v2 * 1.10 and v2 > v3 * 1.05

        # פריצה: close מעל גובה חלון 3 עם נפח פיצוץ
        resistance_vcp = close.iloc[-15:-1].max()
        real_bo = is_real_breakout(
            close, high, resistance_vcp * 0.995,
            vol_ratio_required=BREAKOUT_VOL_MULT,
            volume=volume
        )

        if contracting_price and contracting_vol and real_bo:
            patterns.append("🎯 VCP — צמצום תנודתיות")
            score += 9
    except:
        pass

    return round(min(score, 30), 1), patterns


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Fast Pre-filter
# ══════════════════════════════════════════════════════════════════════════════
def prefilter_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        market_cap = info.get('marketCap', 0) or 0
        if market_cap < MIN_MARKET_CAP:
            return None

        avg_volume = info.get('averageVolume', 0) or 0
        if avg_volume < MIN_AVG_VOLUME:
            return None

        price = info.get('currentPrice') or info.get('regularMarketPrice', 0) or 0
        if price < MIN_PRICE:
            return None

        return {
            'ticker':       ticker,
            'name':         info.get('shortName', ticker),
            'sector':       info.get('sector', 'N/A'),
            'market_cap_b': round(market_cap / 1e9, 1),
            'price':        round(price, 2),
            'pe_ratio':     round(info.get('trailingPE', 0) or 0, 1) or 'N/A',
        }
    except:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Full Pipeline Analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyze_stock(base_info):
    ticker = base_info['ticker']
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 160:
            return None

        close  = hist['Close']
        high   = hist['High']
        low    = hist['Low']
        volume = hist['Volume']

        # ── Small/Mid Cap EMA150 gate ─────────────────────────────────────────
        # מניות עם שווי שוק < $50B חייבות להיות מעל EMA150 כדי להמשיך —
        # מניות קטנות בטרנד שלילי הן מסוכנות מדי.
        # מניות Large Cap (>= $50B) עוברות ישר — הן נסחרות תמיד ויש להן
        # נזילות גבוהה, מנוע A יסנן אותן בהמשך אם צריך.
        if base_info['market_cap_b'] < 50:
            ema150_gate = compute_ema(close, 150).iloc[-1]
            if close.iloc[-1] <= ema150_gate:
                logger.debug(f"{ticker}: Small Cap (${base_info['market_cap_b']}B) below EMA150 — skipped")
                return None

        trend_score,   trend_details = engine_trend(close, high, low, volume)
        if trend_score is None:
            return None

        mom_score, mom_details = engine_momentum(close, high, low, volume)
        if mom_score is None:
            return None

        pattern_score, patterns = engine_patterns(close, high, low, volume)

        final_score = trend_score + mom_score + pattern_score
        if final_score < MIN_FINAL_SCORE:
            return None

        high_52w          = high.iloc[-252:].max() if len(high) >= 252 else high.max()
        pct_from_52w_high = round((close.iloc[-1] / high_52w - 1) * 100, 1)

        return {
            **base_info,
            **trend_details,
            **mom_details,
            'patterns':          patterns,
            'trend_score':       trend_score,
            'mom_score':         mom_score,
            'pattern_score':     pattern_score,
            'final_score':       round(final_score, 1),
            'pct_from_52w_high': pct_from_52w_high,
        }

    except Exception as e:
        logger.debug(f"Error in full analysis {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(tickers):
    logger.info(f"Stage 1: Pre-filtering {len(tickers)} tickers...")
    passed_prefilter = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(prefilter_stock, t): t for t in tickers}
        for i, f in enumerate(as_completed(futures)):
            result = f.result()
            if result:
                passed_prefilter.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"  Pre-filter: {i+1}/{len(tickers)} checked, {len(passed_prefilter)} passed")

    logger.info(f"Stage 1 done: {len(passed_prefilter)} stocks passed pre-filter")

    logger.info(f"Stage 2: Running 3 engines on {len(passed_prefilter)} stocks...")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(analyze_stock, info): info['ticker'] for info in passed_prefilter}
        for i, f in enumerate(as_completed(futures)):
            result = f.result()
            if result:
                results.append(result)
            if (i + 1) % 20 == 0:
                logger.info(f"  Engine scan: {i+1}/{len(passed_prefilter)} analyzed, {len(results)} qualified")

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    logger.info(f"Stage 2 done: {len(df)} stocks passed all 3 engines (score >= {MIN_FINAL_SCORE})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_html_report(df, scan_date, universe_size, prefilter_count):
    top = df.head(TOP_N)

    rows_html = ""
    for i, row in top.iterrows():
        rank    = i + 1
        score   = row['final_score']
        t_score = row['trend_score']
        m_score = row['mom_score']
        p_score = row['pattern_score']

        score_color  = "#22c55e" if score >= 80 else "#f59e0b" if score >= 70 else "#60a5fa"
        patterns_str = ' | '.join(row['patterns']) if row['patterns'] else '—'
        tv_url       = f"https://www.tradingview.com/chart/?symbol={row['ticker']}"

        rows_html += f"""
        <tr>
          <td style="text-align:center;font-weight:700;color:#64748b;font-size:13px;">{rank}</td>
          <td>
            <a href="{tv_url}" target="_blank" style="text-decoration:none;">
              <strong style="color:#60a5fa;font-size:15px;">{row['ticker']}</strong>
            </a><br>
            <span style="color:#94a3b8;font-size:11px;">{row['name']}</span>
          </td>
          <td style="color:#94a3b8;font-size:11px;">{row['sector']}</td>
          <td style="text-align:right;color:#f1f5f9;">${row['price']}</td>
          <td style="text-align:right;color:#94a3b8;">${row['market_cap_b']}B</td>

          <td style="text-align:center;">
            <div style="font-size:11px;color:#94a3b8;">טרנד</div>
            <div style="color:#22c55e;font-weight:700;">{t_score}</div>
          </td>
          <td>
            <div style="font-size:10px;color:#94a3b8;">EMA150: <span style="color:#22c55e;">+{row['ema150_pct']}%</span></div>
            <div style="font-size:10px;color:#94a3b8;">EMA200: {row['above_ema200']} | GX: {row['golden_cross']}</div>
          </td>

          <td style="text-align:center;">
            <div style="font-size:11px;color:#94a3b8;">מומנטום</div>
            <div style="color:#f59e0b;font-weight:700;">{m_score}</div>
          </td>
          <td>
            <div style="font-size:10px;color:#94a3b8;">RSI: <span style="color:#f59e0b;">{row['rsi']}</span> | MACD: {row['macd_bull']}</div>
            <div style="font-size:10px;color:#94a3b8;">נפח: <span style="color:#60a5fa;">{row['vol_ratio']}x</span> | 1M: <span style="color:{'#22c55e' if row['ret_1m'] > 0 else '#f87171'};">{row['ret_1m']}%</span> 3M: <span style="color:{'#22c55e' if row['ret_3m'] > 0 else '#f87171'};">{row['ret_3m']}%</span></div>
            <div style="font-size:10px;color:#f97316;">{row.get('macd_cross','')}</div>
          </td>

          <td style="text-align:center;">
            <div style="font-size:11px;color:#94a3b8;">תבניות</div>
            <div style="color:#a78bfa;font-weight:700;">{p_score}</div>
          </td>
          <td style="font-size:11px;color:#e2e8f0;min-width:160px;">{patterns_str}</td>

          <td style="text-align:center;">
            <div style="background:{score_color};color:#0f172a;font-weight:900;padding:6px 12px;border-radius:20px;font-size:15px;display:inline-block;">{score}</div>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ margin:0; padding:0; background:#0f172a; font-family:'Helvetica Neue',Arial,sans-serif; color:#f1f5f9; }}
  .wrap {{ max-width:1300px; margin:0 auto; padding:28px 16px; }}
  h1 {{ color:#f1f5f9; font-size:26px; margin:0 0 4px; }}
  .sub {{ color:#475569; font-size:13px; margin-bottom:20px; }}
  .stats {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat {{ background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px 16px; font-size:13px; color:#94a3b8; }}
  .stat strong {{ color:#f1f5f9; font-size:16px; display:block; }}
  .engines {{ display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; }}
  .engine {{ border-radius:8px; padding:12px 16px; font-size:12px; flex:1; min-width:180px; }}
  .ea {{ background:#0f2a1a; border:1px solid #22c55e33; }}
  .eb {{ background:#2a1f0a; border:1px solid #f59e0b33; }}
  .ec {{ background:#1a0f2a; border:1px solid #a78bfa33; }}
  .engine-title {{ font-weight:700; font-size:13px; margin-bottom:6px; }}
  .ea .engine-title {{ color:#22c55e; }}
  .eb .engine-title {{ color:#f59e0b; }}
  .ec .engine-title {{ color:#a78bfa; }}
  .engine ul {{ margin:0; padding:0 0 0 14px; color:#94a3b8; line-height:1.7; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ background:#1e293b; color:#475569; padding:8px 6px; text-align:left; font-weight:600; font-size:10px; text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid #334155; }}
  td {{ padding:10px 6px; border-bottom:1px solid #1e293b; vertical-align:top; }}
  tr:hover td {{ background:#ffffff08; }}
  .footer {{ margin-top:24px; color:#334155; font-size:11px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>📈 סורק מניות יומי v4 — Pipeline מקבילי</h1>
  <p class="sub">{scan_date} | NASDAQ & NYSE + Small/Mid Cap</p>

  <div class="stats">
    <div class="stat"><strong>{universe_size}</strong>מניות ביקום</div>
    <div class="stat"><strong>{prefilter_count}</strong>עברו פרה-פילטר</div>
    <div class="stat"><strong>{len(df)}</strong>עברו את 3 המנועים</div>
    <div class="stat"><strong>≥ {MIN_FINAL_SCORE}</strong>ציון מינימום</div>
  </div>

  <div class="engines">
    <div class="engine ea">
      <div class="engine-title">🟢 מנוע A — טרנד (30 נק')</div>
      <ul>
        <li>מחיר מעל EMA150</li>
        <li>מחיר מעל EMA200</li>
        <li>גולדן קרוס EMA20 > EMA50</li>
        <li>שיפוע EMA50 חיובי</li>
      </ul>
    </div>
    <div class="engine eb">
      <div class="engine-title">🟡 מנוע B — מומנטום (40 נק')</div>
      <ul>
        <li>RSI 40–70</li>
        <li>MACD בולי + חצייה טרייה</li>
        <li>Stochastic בולי</li>
        <li>פיצוץ נפח מסחר</li>
        <li>תשואה 1M / 3M</li>
      </ul>
    </div>
    <div class="engine ec">
      <div class="engine-title">🟣 מנוע C — תבניות v4 (30 נק')</div>
      <ul>
        <li>☕ ספל וידית — ולידציה נפח</li>
        <li>🔄 ראש וכתפיים הפוך — ולידציה נפח</li>
        <li>📏 בסיס שטוח — פריצה אמיתית</li>
        <li>📐 משולש עולה — close מעל התנגדות</li>
        <li>🚀 פריצת שיא 52 שבועות — נפח 1.5x</li>
        <li>🚩 דגל שורי — עמוד + דגל שקט</li>
        <li>⚡ תחתית כפולה — ולידציה נפח</li>
        <li>🎯 VCP — צמצום תנודתיות (Minervini)</li>
      </ul>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th><th>מניה</th><th>סקטור</th><th>מחיר</th><th>שווי שוק</th>
        <th>A</th><th>פרטי טרנד</th>
        <th>B</th><th>פרטי מומנטום</th>
        <th>C</th><th>תבניות</th>
        <th>ציון סופי</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="footer">
    ⚠️ לצורכי מידע בלבד — אינו מהווה המלצת השקעה<br>
    Stock Momentum Scanner v4 • {scan_date}
  </div>
</div>
</body>
</html>"""
    return html


def send_email(html_body, scan_date, universe_size, prefilter_count, final_count):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"📈 סורק מניות v4 | {scan_date} | {final_count} מניות עברו את כל המנועים"
    msg['From']    = EMAIL_SENDER
    msg['To']      = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        logger.info(f"✅ Email sent to {EMAIL_RECEIVER}")
    except Exception as e:
        logger.error(f"❌ Email failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    scan_date = datetime.now().strftime("%d/%m/%Y %H:%M")
    logger.info(f"=== Stock Scanner v4 starting: {scan_date} ===")

    tickers = get_stock_universe()
    df      = run_pipeline(tickers)

    if df.empty:
        logger.warning("No stocks passed all 3 engines today.")
        html = f"<h2>סורק מניות v4 — {scan_date}</h2><p>לא נמצאו מניות שעברו את כל 3 המנועים היום.</p>"
        send_email(html, scan_date, len(tickers), 0, 0)
        return

    output_file = f"scan_results_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    html = build_html_report(df, scan_date, len(tickers), len(df))
    send_email(html, scan_date, len(tickers), len(df), min(TOP_N, len(df)))

    logger.info(f"=== Scanner v4 finished: {len(df)} stocks qualified ===")


if __name__ == "__main__":
    main()
