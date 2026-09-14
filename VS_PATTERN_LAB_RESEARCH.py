# VS PATTERN LAB — Research Engine
# Single-file VS FLOW-style research build. No broker/API/alerts/auto-trading.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf


# ===== pattern_meta.py =====
# Explicit metadata prevents direction from being guessed from pattern names.
PATTERN_META = {}

def add(names, direction, category="Candlestick", family="standard"):
    for name in names:
        PATTERN_META[name] = {"direction": direction, "category": category, "family": family}

add(["Hammer","Inverted Hammer","Dragonfly Doji","Bullish Marubozu","Bullish Pin Bar",
     "Bullish Engulfing","Bullish Harami","Bullish Harami Cross","Piercing Line","Tweezer Bottom",
     "Bullish Kicker","Bullish Belt Hold","Bullish Counterattack","Bullish Separating Lines",
     "Morning Star","Morning Doji Star","Three White Soldiers","Three Inside Up","Three Outside Up",
     "Abandoned Baby Bullish","Bullish Tri-Star","Bullish Tasuki Gap","Bullish Three Line Strike",
     "Rising Three Methods","Bullish Breakaway","Bullish Hikkake","Island Reversal Bullish"], "Bullish")
add(["Hanging Man","Shooting Star","Gravestone Doji","Bearish Marubozu","Bearish Pin Bar",
     "Bearish Engulfing","Bearish Harami","Bearish Harami Cross","Dark Cloud Cover","Tweezer Top",
     "Bearish Kicker","Bearish Belt Hold","Bearish Counterattack","Bearish Separating Lines",
     "Evening Star","Evening Doji Star","Three Black Crows","Three Inside Down","Three Outside Down",
     "Abandoned Baby Bearish","Bearish Tri-Star","Bearish Tasuki Gap","Bearish Three Line Strike",
     "Falling Three Methods","Bearish Breakaway","Bearish Hikkake","Island Reversal Bearish"], "Bearish")
add(["Doji","Spinning Top","Counterattack","Matching High","Matching Low","Homing Pigeon","Descending Hawk"], "Neutral")

CHART_META = {
    "Head-and-Shoulders": "Bearish", "Inverse Head-and-Shoulders":"Bullish",
    "Double Top":"Bearish", "Double Bottom":"Bullish", "Triple Top":"Bearish", "Triple Bottom":"Bullish",
    "Ascending Triangle":"Bullish", "Descending Triangle":"Bearish", "Symmetrical Triangle":"Neutral",
    "Rising Wedge":"Bearish", "Falling Wedge":"Bullish", "Rectangle":"Neutral",
    "Bull Flag":"Bullish", "Bear Flag":"Bearish", "Bull Pennant":"Bullish", "Bear Pennant":"Bearish",
    "Cup and Handle":"Bullish", "Inverse Cup and Handle":"Bearish", "Rounding Bottom":"Bullish",
    "Rounding Top":"Bearish", "Broadening Formation":"Neutral", "V Pattern":"Bullish", "Inverse V Pattern":"Bearish",
    "Higher Highs & Higher Lows":"Bullish", "Lower Highs & Lower Lows":"Bearish"
}
for k,v in CHART_META.items():
    PATTERN_META[k] = {"direction":v,"category":"Chart Pattern","family":"geometry"}

def meta(name):
    if name in PATTERN_META:
        return PATTERN_META[name]
    return {"direction":"Neutral","category":"Unknown","family":"heuristic"}


# ===== candles.py =====


def _body(df):
    return (df["Close"] - df["Open"]).abs()

def _range(df):
    return (df["High"] - df["Low"]).replace(0, np.nan)

def _upper(df):
    return df["High"] - df[["Open","Close"]].max(axis=1)

def _lower(df):
    return df[["Open","Close"]].min(axis=1) - df["Low"]

def detect_candles(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    body, rng, up, lo = _body(x), _range(x), _upper(x), _lower(x)
    bull = x["Close"] > x["Open"]
    bear = x["Close"] < x["Open"]
    small = body <= rng * 0.25
    long_body = body >= rng * 0.60
    doji = body <= rng * 0.10
    long_lower = lo >= body * 2
    long_upper = up >= body * 2

    # One-candle patterns
    x["Hammer"] = long_lower & (up <= body * 0.6) & (body <= rng * 0.40)
    x["Inverted Hammer"] = long_upper & (lo <= body * 0.6) & (body <= rng * 0.40)
    x["Hanging Man"] = x["Hammer"]
    x["Shooting Star"] = x["Inverted Hammer"]
    x["Doji"] = doji
    x["Dragonfly Doji"] = doji & (up <= rng * 0.08) & (lo >= rng * 0.60)
    x["Gravestone Doji"] = doji & (lo <= rng * 0.08) & (up >= rng * 0.60)
    x["Spinning Top"] = small & (up >= body) & (lo >= body)
    x["Bullish Marubozu"] = bull & long_body & (up <= rng * 0.05) & (lo <= rng * 0.05)
    x["Bearish Marubozu"] = bear & long_body & (up <= rng * 0.05) & (lo <= rng * 0.05)
    x["Bullish Pin Bar"] = bull & long_lower & (lo / rng >= 0.55)
    x["Bearish Pin Bar"] = bear & long_upper & (up / rng >= 0.55)
    x["Long White Candle"] = bull & long_body
    x["Long Black Candle"] = bear & long_body

    # Two-candle patterns
    po, pc = x["Open"].shift(1), x["Close"].shift(1)
    ph, pl = x["High"].shift(1), x["Low"].shift(1)
    pbull, pbear = pc > po, pc < po
    pbody = (pc-po).abs()

    x["Bullish Engulfing"] = pbear & bull & (x["Open"] <= pc) & (x["Close"] >= po) & (body > pbody)
    x["Bearish Engulfing"] = pbull & bear & (x["Open"] >= pc) & (x["Close"] <= po) & (body > pbody)
    x["Bullish Harami"] = pbear & bull & (x["Open"] >= pc) & (x["Close"] <= po) & (body < pbody)
    x["Bearish Harami"] = pbull & bear & (x["Open"] <= pc) & (x["Close"] >= po) & (body < pbody)
    x["Bullish Harami Cross"] = x["Bullish Harami"] & doji
    x["Bearish Harami Cross"] = x["Bearish Harami"] & doji
    x["Piercing Line"] = pbear & bull & (x["Close"] > (po+pc)/2) & (x["Close"] < po)
    x["Dark Cloud Cover"] = pbull & bear & (x["Close"] < (po+pc)/2) & (x["Close"] > po)
    x["Tweezer Bottom"] = (pl / x["Low"] - 1).abs() < 0.003
    x["Tweezer Top"] = (ph / x["High"] - 1).abs() < 0.003
    x["Bullish Kicker"] = pbear & bull & (x["Open"] > ph)
    x["Bearish Kicker"] = pbull & bear & (x["Open"] < pl)
    x["Bullish Belt Hold"] = bull & (x["Open"] <= x["Low"]*1.002) & (body/rng > 0.65)
    x["Bearish Belt Hold"] = bear & (x["Open"] >= x["High"]*0.998) & (body/rng > 0.65)
    x["Bullish Counterattack"] = pbear & bull & (abs(x["Close"]-pc) / x["Close"] < 0.003)
    x["Bearish Counterattack"] = pbull & bear & (abs(x["Close"]-pc) / x["Close"] < 0.003)
    x["Bullish Separating Lines"] = pbear & bull & (abs(x["Open"]-po) / x["Open"] < 0.003) & (x["Close"] > pc)
    x["Bearish Separating Lines"] = pbull & bear & (abs(x["Open"]-po) / x["Open"] < 0.003) & (x["Close"] < pc)
    x["Matching Low"] = pbear & bull & (abs(x["Close"]-pc)/x["Close"] < 0.003)
    x["Matching High"] = pbull & bear & (abs(x["Close"]-pc)/x["Close"] < 0.003)

    # Three-candle patterns
    p2o, p2c = x["Open"].shift(2), x["Close"].shift(2)
    p2h, p2l = x["High"].shift(2), x["Low"].shift(2)
    p2bull, p2bear = p2c > p2o, p2c < p2o
    mid_small = ((x["Open"].shift(1)-x["Close"].shift(1)).abs() <=
                 (x["High"].shift(1)-x["Low"].shift(1))*0.30)
    x["Morning Star"] = p2bear & mid_small & bull & (x["Close"] > (p2o+p2c)/2)
    x["Evening Star"] = p2bull & mid_small & bear & (x["Close"] < (p2o+p2c)/2)
    x["Morning Doji Star"] = x["Morning Star"] & x["Doji"].shift(1)
    x["Evening Doji Star"] = x["Evening Star"] & x["Doji"].shift(1)
    x["Three White Soldiers"] = bull & bull.shift(1) & bull.shift(2) & (x["Close"] > x["Close"].shift(1)) & (x["Close"].shift(1) > x["Close"].shift(2))
    x["Three Black Crows"] = bear & bear.shift(1) & bear.shift(2) & (x["Close"] < x["Close"].shift(1)) & (x["Close"].shift(1) < x["Close"].shift(2))
    x["Three Inside Up"] = x["Bullish Harami"].shift(1) & bull & (x["Close"] > x["High"].shift(1))
    x["Three Inside Down"] = x["Bearish Harami"].shift(1) & bear & (x["Close"] < x["Low"].shift(1))
    x["Three Outside Up"] = x["Bullish Engulfing"].shift(1) & bull & (x["Close"] > x["High"].shift(1))
    x["Three Outside Down"] = x["Bearish Engulfing"].shift(1) & bear & (x["Close"] < x["Low"].shift(1))
    x["Bullish Tri-Star"] = x["Doji"] & x["Doji"].shift(1) & x["Doji"].shift(2)
    x["Bearish Tri-Star"] = x["Bullish Tri-Star"]
    x["Bullish Abandoned Baby"] = p2bear & x["Doji"].shift(1) & bull & (x["Low"].shift(1) > p2h) & (x["Low"] > x["High"].shift(1))
    x["Bearish Abandoned Baby"] = p2bull & x["Doji"].shift(1) & bear & (x["High"].shift(1) < p2l) & (x["High"] < x["Low"].shift(1))
    x["Upside Tasuki Gap"] = bull.shift(2) & bull.shift(1) & bear & (x["Open"].shift(1) > x["High"].shift(2)) & (x["Open"] < x["Close"].shift(1))
    x["Downside Tasuki Gap"] = bear.shift(2) & bear.shift(1) & bull & (x["Open"].shift(1) < x["Low"].shift(2)) & (x["Open"] > x["Close"].shift(1))
    x["Rising Three Methods"] = bear.shift(3) & bull.shift(2) & bull.shift(1) & bull & (x["Close"] > x["Close"].shift(3))
    x["Falling Three Methods"] = bull.shift(3) & bear.shift(2) & bear.shift(1) & bear & (x["Close"] < x["Close"].shift(3))
    x["Bullish Three Line Strike"] = bull & bear.shift(1) & bear.shift(2) & bear.shift(3) & (x["Close"] > x["Open"].shift(3))
    x["Bearish Three Line Strike"] = bear & bull.shift(1) & bull.shift(2) & bull.shift(3) & (x["Close"] < x["Open"].shift(3))

    # Generic aliases requested by common screeners
    x["Pin Bar"] = x["Bullish Pin Bar"] | x["Bearish Pin Bar"]
    x["Hammer At Downtrend"] = x["Hammer"]
    x["Inverted Hammer At Downtrend"] = x["Inverted Hammer"]
    x["Shooting Star At Uptrend"] = x["Shooting Star"]
    x["Hanging Man At Uptrend"] = x["Hanging Man"]

    pattern_cols = [c for c in x.columns if c not in df.columns]
    return x[pattern_cols].fillna(False).astype(bool)


# ===== chart_patterns.py =====


def pivots(df, window=3):
    h, l = df["High"], df["Low"]
    ph = (h == h.rolling(2*window+1, center=True).max()).fillna(False)
    pl = (l == l.rolling(2*window+1, center=True).min()).fillna(False)
    return ph, pl

def _last_points(series, mask, n=6):
    return series[mask].tail(n)

def detect_chart_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        return {}
    ph, pl = pivots(df, 3)
    highs = _last_points(df["High"], ph, 8)
    lows = _last_points(df["Low"], pl, 8)
    close = float(df["Close"].iloc[-1])
    out = {}

    def add(name, direction, score, level=None, detail=""):
        out[name] = {"direction": direction, "score": int(max(0,min(100,score))),
                     "level": None if level is None else float(level), "detail": detail}

    # Double / triple top-bottom using recent pivot similarity
    if len(highs) >= 2:
        a,b = highs.iloc[-2], highs.iloc[-1]
        tol = close*0.025
        if abs(a-b) <= tol:
            add("Double Top", "Bearish", 72, min(a,b), "Two recent swing highs are within tolerance.")
    if len(lows) >= 2:
        a,b = lows.iloc[-2], lows.iloc[-1]
        tol = close*0.025
        if abs(a-b) <= tol:
            add("Double Bottom", "Bullish", 72, max(a,b), "Two recent swing lows are within tolerance.")
    if len(highs) >= 3:
        a,b,c = highs.iloc[-3:]
        if max(a,b,c)-min(a,b,c) <= close*0.035:
            add("Triple Top", "Bearish", 78, min(a,b,c), "Three aligned swing highs.")
    if len(lows) >= 3:
        a,b,c = lows.iloc[-3:]
        if max(a,b,c)-min(a,b,c) <= close*0.035:
            add("Triple Bottom", "Bullish", 78, max(a,b,c), "Three aligned swing lows.")

    # H&S / inverse H&S
    if len(highs) >= 3:
        a,b,c = highs.iloc[-3:]
        if b > a*1.01 and b > c*1.01 and abs(a-c)/close < 0.045:
            add("Head-and-Shoulders Top", "Bearish", 86, min(a,c), "Left/right shoulders with a higher head.")
    if len(lows) >= 3:
        a,b,c = lows.iloc[-3:]
        if b < a*0.99 and b < c*0.99 and abs(a-c)/close < 0.045:
            add("Inverse Head-and-Shoulders", "Bullish", 86, max(a,c), "Left/right shoulders with a lower head.")

    # Trend / range structures
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    slope20 = float(sma20.iloc[-1] - sma20.iloc[-10])
    slope50 = float(sma50.iloc[-1] - sma50.iloc[-10])
    recent = df.tail(30)
    hi, lo = recent["High"].max(), recent["Low"].min()
    width = hi-lo

    if slope20 > 0 and slope50 > 0:
        add("Higher Highs & Higher Lows", "Bullish", 68, lo, "Short and medium trend slopes are rising.")
    if slope20 < 0 and slope50 < 0:
        add("Lower Highs & Lower Lows", "Bearish", 68, hi, "Short and medium trend slopes are falling.")

    # Triangles / wedges from pivot trend slopes
    if len(highs) >= 3 and len(lows) >= 3:
        hx = np.arange(len(highs))
        lx = np.arange(len(lows))
        hs = np.polyfit(hx, highs.values, 1)[0]
        ls = np.polyfit(lx, lows.values, 1)[0]
        if hs < 0 and ls > 0:
            add("Symmetrical Triangle", "Neutral", 76, close, "Falling highs and rising lows.")
        if abs(hs) < close*0.003 and ls > close*0.003:
            add("Ascending Triangle", "Bullish", 79, float(highs.mean()), "Flat-ish resistance with rising lows.")
        if hs < -close*0.003 and abs(ls) < close*0.003:
            add("Descending Triangle", "Bearish", 79, float(lows.mean()), "Falling highs with flat-ish support.")
        if hs > 0 and ls > 0 and hs < ls:
            add("Rising Wedge", "Bearish", 75, close, "Both boundaries rise while converging.")
        if hs < 0 and ls < 0 and abs(hs) < abs(ls):
            add("Falling Wedge", "Bullish", 75, close, "Both boundaries fall while converging.")

    # Rectangle / range
    if width > 0:
        closes = recent["Close"]
        if closes.max()-closes.min() <= width*0.65:
            add("Rectangle / Range", "Neutral", 65, hi, "Price is compressing inside a defined range.")

    # Cup / rounding approximations
    if len(df) >= 100:
        q = df["Close"].tail(100).to_numpy()
        left, mid, right = q[:25].mean(), q[40:65].mean(), q[-25:].mean()
        if mid < left*0.94 and abs(left-right)/close < 0.06:
            add("Rounding Bottom / Cup", "Bullish", 74, max(left,right), "U-shaped recovery with similar rim areas.")
        if mid > left*1.06 and abs(left-right)/close < 0.06:
            add("Rounding Top", "Bearish", 74, min(left,right), "Rounded distribution with similar rim areas.")

    # Flags / pennants approximation after impulse
    if len(df) >= 40:
        impulse = (df["Close"].iloc[-40:-25].iloc[-1] / df["Close"].iloc[-40:-25].iloc[0] - 1)
        cons = df["Close"].tail(15)
        cons_width = (cons.max()-cons.min())/max(cons.mean(),1e-9)
        if abs(impulse) > 0.08 and cons_width < 0.08:
            name = "Bullish Flag" if impulse > 0 else "Bearish Flag"
            direction = "Bullish" if impulse > 0 else "Bearish"
            add(name, direction, 73, close, "Impulse followed by compact consolidation.")

    # Broadening approximation
    if len(highs) >= 4 and len(lows) >= 4:
        hspread = abs(highs.iloc[-1]-highs.iloc[-3])
        lspread = abs(lows.iloc[-1]-lows.iloc[-3])
        if hspread > close*0.03 and lspread > close*0.03:
            add("Broadening Formation", "Neutral", 67, close, "Expanding swing range.")

    return out


# ===== context.py =====

def add_context(df):
    x=df.copy()
    x["SMA20"]=x.Close.rolling(20).mean()
    x["SMA50"]=x.Close.rolling(50).mean()
    x["SMA200"]=x.Close.rolling(200).mean()
    prev=x.Close.shift(1)
    tr=pd.concat([(x.High-x.Low),(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1)
    x["ATR14"]=tr.rolling(14).mean()
    x["VOL20"]=x.Volume.rolling(20).mean()
    x["VOL_RATIO"]=x.Volume/x.VOL20.replace(0,np.nan)
    x["ROC20"]=x.Close.pct_change(20)*100
    x["HH20"]=x.High.rolling(20).max()
    x["LL20"]=x.Low.rolling(20).min()
    return x

def score_context(df,direction,raw):
    x=add_context(df); r=x.iloc[-1]; score=float(raw); notes=[]
    if pd.notna(r.SMA20) and pd.notna(r.SMA50):
        bull=r.Close>r.SMA20>r.SMA50; bear=r.Close<r.SMA20<r.SMA50
        if (direction=="Bullish" and bull) or (direction=="Bearish" and bear): score+=8; notes.append("HTF-style trend aligned")
        elif direction!="Neutral": score-=5; notes.append("trend mixed")
    if pd.notna(r.VOL_RATIO):
        if r.VOL_RATIO>=1.5: score+=7; notes.append("volume expansion")
        elif r.VOL_RATIO<0.7: score-=3; notes.append("low volume")
    if pd.notna(r.ROC20) and direction!="Neutral":
        if (direction=="Bullish" and r.ROC20>0) or (direction=="Bearish" and r.ROC20<0): score+=4; notes.append("momentum aligned")
    atr_pct=(r.ATR14/r.Close*100) if pd.notna(r.ATR14) and r.Close else np.nan
    if pd.notna(atr_pct): notes.append(f"ATR {atr_pct:.2f}%")
    return int(np.clip(round(score),0,100)), "; ".join(notes)


# ===== geometry.py =====

def pivot_points(df,left=3,right=3):
    h=df.High.to_numpy(); l=df.Low.to_numpy(); hi=[]; lo=[]
    for i in range(left,len(df)-right):
        if h[i]>=h[i-left:i+right+1].max(): hi.append((i,float(h[i])))
        if l[i]<=l[i-left:i+right+1].min(): lo.append((i,float(l[i])))
    return hi,lo

def relerr(a,b): return abs(a-b)/max(abs(b),1e-12)

def detect_geometry(df):
    if len(df)<80:return []
    highs,lows=pivot_points(df); last=float(df.Close.iloc[-1]); out=[]
    def add(name,direction,score,confirmation,invalidation,level=None):
        out.append({"pattern":name,"direction":direction,"score":score,"confirmation":confirmation,
                    "invalidation":invalidation,"confirmation_level":level,"invalidation_level":invalidation if isinstance(invalidation,(int,float)) else None})
    if len(highs)>=3 and len(lows)>=2:
        vals=[x[1] for x in highs[-3:]]
        if vals[1]>vals[0] and vals[1]>vals[2] and relerr(vals[0],vals[2])<=.05:
            neck=min(x[1] for x in lows[-2:]); add("Head-and-Shoulders","Bearish",92,"Neckline broken" if last<neck else "Await neckline break",max(vals),neck)
    if len(lows)>=3 and len(highs)>=2:
        vals=[x[1] for x in lows[-3:]]
        if vals[1]<vals[0] and vals[1]<vals[2] and relerr(vals[0],vals[2])<=.05:
            neck=max(x[1] for x in highs[-2:]); add("Inverse Head-and-Shoulders","Bullish",92,"Neckline broken" if last>neck else "Await neckline break",min(vals),neck)
    if len(highs)>=2 and relerr(highs[-2][1],highs[-1][1])<=.025:
        neck=min(x[1] for x in lows[-2:]) if len(lows)>=2 else min(highs[-2][1],highs[-1][1])
        add("Double Top","Bearish",90 if last<neck else 76,"Confirmed breakdown" if last<neck else "Await breakdown",max(x[1] for x in highs[-2:]),neck)
    if len(lows)>=2 and relerr(lows[-2][1],lows[-1][1])<=.025:
        neck=max(x[1] for x in highs[-2:]) if len(highs)>=2 else max(lows[-2][1],lows[-1][1])
        add("Double Bottom","Bullish",90 if last>neck else 76,"Confirmed breakout" if last>neck else "Await breakout",min(x[1] for x in lows[-2:]),neck)
    if len(highs)>=4 and len(lows)>=4:
        hs=np.polyfit(np.arange(4),[x[1] for x in highs[-4:]],1)[0]; ls=np.polyfit(np.arange(4),[x[1] for x in lows[-4:]],1)[0]; tol=last*.0025
        if abs(hs)<=tol and ls>tol:
            level=max(x[1] for x in highs[-4:]); add("Ascending Triangle","Bullish",84,"Resistance breakout" if last>level else "Await resistance breakout",min(x[1] for x in lows[-4:]),level)
        elif hs<-tol and abs(ls)<=tol:
            level=min(x[1] for x in lows[-4:]); add("Descending Triangle","Bearish",84,"Support breakdown" if last<level else "Await support breakdown",max(x[1] for x in highs[-4:]),level)
        elif hs<-tol and ls>tol:
            add("Symmetrical Triangle","Neutral",80,"Await boundary break","Boundary break required",None)
    return out


# ===== lifecycle.py =====

def classify_lifecycle(df, direction, confirmation_level=None, invalidation_level=None):
    """Research lifecycle: Formation -> Developing -> Confirmed -> Retest -> Failed -> Invalidated."""
    if df.empty or len(df)<20: return "Formation",35
    close=float(df.Close.iloc[-1]); recent=df.tail(8)
    atr=float((df.High-df.Low).rolling(14).mean().iloc[-1] or 0)
    tol=max(atr*0.35, close*0.001)
    if invalidation_level is not None:
        inv=float(invalidation_level)
        if (direction=="Bullish" and close<inv-tol) or (direction=="Bearish" and close>inv+tol): return "Invalidated",10
    if confirmation_level is None:
        # A pattern with no explicit trigger is still only developing.
        return ("Developing",55) if len(recent)>=4 else ("Formation",40)
    lvl=float(confirmation_level)
    if direction=="Bullish":
        if close>lvl+tol:
            # Retest = recent candles came back near trigger after breaking it.
            after_break=recent[recent.Close>lvl]
            if len(after_break)>=2 and recent.Low.min()<=lvl+tol: return "Retest",86
            return "Confirmed",90
        if close<lvl-tol*1.5: return "Failed",20
        return "Developing",65
    if direction=="Bearish":
        if close<lvl-tol:
            after_break=recent[recent.Close<lvl]
            if len(after_break)>=2 and recent.High.max()>=lvl-tol: return "Retest",86
            return "Confirmed",90
        if close>lvl+tol*1.5: return "Failed",20
        return "Developing",65
    return "Developing",55


# ===== data.py =====

TIMEFRAME_CONFIG = {
    "5M": {"interval": "5m", "period": "5d"},
    "15M": {"interval": "15m", "period": "60d"},
    "1H": {"interval": "60m", "period": "730d"},
    "4H": {"interval": "60m", "period": "730d"},
    "1D": {"interval": "1d", "period": "10y"},
    "1W": {"interval": "1wk", "period": "max"},
    "1MN": {"interval": "1mo", "period": "max"},
}

def normalize_symbol(symbol, market="India"):
    s = str(symbol).strip().upper()
    if s.startswith("^") or "." in s:
        return s
    return s + ".NS" if market == "India" else s

def _resample_4h(df):
    if df.empty:
        return df
    x = df.copy()
    # Exchange-session boundaries are not assumed; this is a research approximation.
    out = x.resample("4h", origin="start_day").agg({
        "Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"
    }).dropna(subset=["Open","High","Low","Close"])
    return out

def fetch_ohlcv(symbol, timeframe, market="India"):
    cfg = TIMEFRAME_CONFIG[timeframe]
    ticker = normalize_symbol(symbol, market)
    df = yf.download(ticker, interval=cfg["interval"], period=cfg["period"],
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[cols].dropna().copy()
    df.index = pd.to_datetime(df.index)
    if timeframe == "4H":
        df = _resample_4h(df)
    return df

INDIA_DEFAULT = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","BHARTIARTL","ITC",
    "LT","AXISBANK","KOTAKBANK","HINDUNILVR","MARUTI","SUNPHARMA","BAJFINANCE",
    "ADANIENT","TITAN","ULTRACEMCO","ASIANPAINT","HCLTECH"
]
GLOBAL_DEFAULT = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","NFLX",
    "JPM","WMT","ORCL","COST","INTC","QCOM","CRM","ADBE","UBER","COIN"
]
INDEX_SYMBOLS = {
    "NIFTY 50":"^NSEI","BANK NIFTY":"^NSEBANK","S&P 500":"^GSPC","NASDAQ":"^IXIC",
    "DOW JONES":"^DJI","NIKKEI":"^N225","HANG SENG":"^HSI"
}


# ===== research_engine.py =====

def pattern_events(df):
    """Create point-in-time pattern events. Only information available at each candle is used."""
    rows=[]
    candles=detect_candles(df)
    for i in range(len(df)):
        for p in candles.columns:
            try: hit=bool(candles[p].iloc[i])
            except Exception: hit=False
            if hit:
                m=meta(p); rows.append({"Index":i,"Pattern":p,"Category":m["category"],"Direction":m["direction"],"RawScore":72 if m["direction"]!="Neutral" else 62})
    # Geometry is intentionally evaluated on expanding history, preventing future leakage.
    for i in range(80,len(df)):
        sub=df.iloc[:i+1]
        for v in detect_geometry(sub):
            rows.append({"Index":i,"Pattern":v["pattern"],"Category":"Geometry","Direction":v["direction"],"RawScore":v["score"]})
    return pd.DataFrame(rows)

def evaluate_pattern(df, pattern_name, direction=None, horizon=10):
    ev=pattern_events(df)
    if ev.empty:return pd.DataFrame(),{}
    ev=ev[ev.Pattern==pattern_name].copy()
    if direction:ev=ev[ev.Direction==direction]
    trades=[]
    for _,e in ev.iterrows():
        i=int(e.Index); entry_i=i+1; exit_i=entry_i+horizon
        if exit_i>=len(df):continue
        entry=float(df.Open.iloc[entry_i]); exit_=float(df.Close.iloc[exit_i]); ret=(exit_/entry-1)*100
        if e.Direction=="Bearish":ret=-ret
        trades.append({**e.to_dict(),"EntryIndex":entry_i,"ExitIndex":exit_i,"ReturnPct":ret})
    out=pd.DataFrame(trades)
    if out.empty:return out,{"signals":0}
    r=out.ReturnPct
    wins=r[r>0]; losses=r[r<0]
    return out,{"signals":len(r),"win_rate_pct":round((r>0).mean()*100,2),"avg_return_pct":round(r.mean(),3),"median_return_pct":round(r.median(),3),"profit_factor":round(wins.sum()/abs(losses.sum()),3) if len(losses) else float("inf"),"max_loss_pct":round(r.min(),3)}


# ===== research.py =====

def signal_series(df, direction, horizon=10, entry_delay=1):
    close=df.Close
    if direction=="Bullish": signal=close>close.shift(1)
    elif direction=="Bearish": signal=close<close.shift(1)
    else: signal=pd.Series(False,index=df.index)
    rows=[]
    for i in np.flatnonzero(signal.to_numpy()):
        entry_i=i+entry_delay; exit_i=entry_i+horizon
        if exit_i>=len(df): continue
        entry=float(df.Open.iloc[entry_i]); exit_=float(df.Close.iloc[exit_i])
        ret=(exit_/entry-1)*100
        if direction=="Bearish": ret=-ret
        rows.append({"SignalIndex":i,"EntryIndex":entry_i,"ExitIndex":exit_i,"Entry":entry,"Exit":exit_,"ReturnPct":ret})
    return pd.DataFrame(rows)

def backtest_fixed_horizon(df,direction,horizon=10):
    trades=signal_series(df,direction,horizon=horizon,entry_delay=1)
    if trades.empty: return trades, report(trades.ReturnPct if "ReturnPct" in trades else pd.Series(dtype=float))
    return trades,report(trades.ReturnPct)

def report(returns):
    r=pd.Series(returns,dtype=float).dropna()
    if r.empty:return {"signals":0,"win_rate_pct":0.0,"avg_return_pct":0.0,"median_return_pct":0.0,"profit_factor":0.0,"max_loss_pct":0.0,"max_drawdown_pct":0.0}
    eq=(1+r/100).cumprod(); peak=eq.cummax(); dd=(eq/peak-1)*100
    wins=r[r>0]; losses=r[r<0]
    gp=wins.sum(); gl=abs(losses.sum())
    return {"signals":len(r),"win_rate_pct":float((r>0).mean()*100),"avg_return_pct":float(r.mean()),"median_return_pct":float(r.median()),"profit_factor":float(gp/gl) if gl else float("inf"),"max_loss_pct":float(r.min()),"max_drawdown_pct":float(dd.min())}

def walk_forward(df,direction,horizon=10,train_ratio=.7):
    n=len(df); split=max(int(n*train_ratio),50)
    train=df.iloc[:split]; test=df.iloc[split:]
    _,a=backtest_fixed_horizon(train,direction,horizon)
    _,b=backtest_fixed_horizon(test,direction,horizon)
    return {"train":a,"out_of_sample":b,"split_index":split}


# ===== scanner.py =====

def scan_symbol(symbol,timeframe,market):
    df=fetch_ohlcv(symbol,timeframe,market)
    if df.empty or len(df)<30: return []
    rows=[]
    candles=detect_candles(df)
    for p in candles.columns:
        if bool(candles[p].iloc[-1]):
            m=meta(p); direction=m["direction"]
            raw=72 if direction!="Neutral" else 62
            score,note=score_context(df,direction,raw)
            status,life=classify_lifecycle(df,direction)
            rows.append({"Symbol":symbol,"Pattern":p,"Category":"Candlestick","TF":timeframe,
                         "Direction":direction,"Status":status,"Score":score,"LifecycleScore":life,"Context":note})
    # Existing broad detector
    for p,v in detect_chart_patterns(df).items():
        direction=v.get("direction",meta(p)["direction"])
        score,note=score_context(df,direction,v.get("score",70))
        status,life=classify_lifecycle(df,direction,v.get("level"),v.get("invalidation"))
        rows.append({"Symbol":symbol,"Pattern":p,"Category":"Chart Pattern","TF":timeframe,
                     "Direction":direction,"Status":status,"Score":score,"LifecycleScore":life,
                     "Context":v.get("detail","")+(("; "+note) if note else "")})
    # Deterministic geometry engine
    for v in detect_geometry(df):
        direction=v["direction"]
        score,note=score_context(df,direction,v["score"])
        status,life=classify_lifecycle(df,direction,v.get("confirmation_level"),v.get("invalidation_level"))
        rows.append({"Symbol":symbol,"Pattern":v["pattern"],"Category":"Geometry","TF":timeframe,
                     "Direction":direction,"Status":status,"Score":score,"LifecycleScore":life,
                     "Context":v.get("confirmation","")+(("; "+note) if note else "")})
    return rows

def scan_universe(symbols,timeframe,market):
    rows=[]
    for s in symbols:
        try: rows.extend(scan_symbol(s,timeframe,market))
        except Exception as e: rows.append({"Symbol":s,"Pattern":"ERROR","Category":"System","TF":timeframe,"Direction":"Neutral","Status":"Error","Score":0,"LifecycleScore":0,"Context":str(e)})
    return pd.DataFrame(rows)


# ===== mtf.py =====

TFS=["5M","15M","1H","4H","1D","1W","1MN"]
WEIGHTS={"5M":1,"15M":1,"1H":2,"4H":2,"1D":3,"1W":3,"1MN":2}

def mtf_analysis(symbol,market):
    rows=[]
    for tf in TFS:
        try:
            found=scan_symbol(symbol,tf,market)
            bullish=[r for r in found if r["Direction"]=="Bullish"]
            bearish=[r for r in found if r["Direction"]=="Bearish"]
            b=max([r["Score"] for r in bullish],default=0); s=max([r["Score"] for r in bearish],default=0)
            if b>=s+5 and b>=65: bias="Bullish"
            elif s>=b+5 and s>=65: bias="Bearish"
            else: bias="Neutral"
            rows.append({"TF":tf,"Bias":bias,"BullScore":b,"BearScore":s,"Patterns":len(found)})
        except Exception:
            rows.append({"TF":tf,"Bias":"Error","BullScore":0,"BearScore":0,"Patterns":0})
    df=pd.DataFrame(rows)
    score_b=sum(WEIGHTS[r.TF]*r.BullScore for _,r in df.iterrows())/sum(WEIGHTS.values())
    score_s=sum(WEIGHTS[r.TF]*r.BearScore for _,r in df.iterrows())/sum(WEIGHTS.values())
    if score_b>=score_s+5: final="Bullish"
    elif score_s>=score_b+5: final="Bearish"
    else: final="Neutral"
    return df, {"FinalBias":final,"BullWeighted":round(score_b,1),"BearWeighted":round(score_s,1)}


# ===== UI =====

st.set_page_config(page_title="VS PATTERN LAB", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 70% -10%,rgba(0,157,255,.13),transparent 35%),linear-gradient(180deg,#03070d 0%,#050b13 55%,#02060b 100%);color:#f3f7fb}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1600px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#050b14 0%,#06111d 100%);border-right:1px solid #12344d}
.vs-brand{text-align:center;padding:4px 0 16px}.vs-brand-title{font-size:20px;font-weight:900;letter-spacing:.5px}.vs-brand-title span{color:#13d9ff}.vs-brand-sub{font-size:9px;color:#7694a8;letter-spacing:1.5px;margin-top:5px}
.side-section{font-size:10px;color:#6f8da4;font-weight:800;letter-spacing:1.2px;margin:8px 2px}
.hero{border:1px solid #124466;border-radius:18px;background:linear-gradient(135deg,rgba(8,30,48,.96),rgba(4,13,23,.96));padding:21px 25px 18px;box-shadow:0 12px 40px rgba(0,0,0,.28);position:relative;overflow:hidden}.hero:after{content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:linear-gradient(90deg,#10dcff,#ff29d5,#ff5c2f,#ffd52f)}
.hero-top{display:flex;align-items:center;justify-content:space-between;gap:20px}.hero-title{font-size:31px;font-weight:950;letter-spacing:.4px}.hero-title span{color:#13d9ff}.hero-sub{color:#7190a8;font-size:11px;margin-top:6px}.pill{border:1px solid #1b5778;border-radius:999px;padding:7px 13px;color:#dceaf4;font-size:10px;font-weight:800;background:#071522}
.section-title{font-size:17px;font-weight:900;letter-spacing:.4px;margin:21px 0 10px}.section-title small{font-size:10px;color:#6d8ba0;font-weight:500;margin-left:7px}
.card,.regime{border:1px solid #123b57;border-radius:15px;background:linear-gradient(145deg,#071927,#06121e);padding:16px 17px;min-height:104px;box-shadow:0 8px 24px rgba(0,0,0,.18)}
.regime{border-color:#135d58;background:linear-gradient(145deg,#06241f,#07141c)}.regime-tag,.card-label{font-size:9px;color:#6d8ba0;font-weight:800;letter-spacing:1px}.regime-tag{color:#19e68b}.regime-value{font-size:22px;font-weight:950;margin-top:7px}.card-value{font-size:25px;font-weight:900;margin-top:7px}.card-note,.regime-note{font-size:9px;color:#6f8ca3;margin-top:6px}.cyan{color:#16ddff}.green{color:#19e68b}.red{color:#ff5960}.yellow{color:#ffd04a}
.stTabs [data-baseweb="tab-list"]{gap:5px;border-bottom:1px solid #12344c}.stTabs [data-baseweb="tab"]{height:38px;padding:0 14px;color:#7894a8;font-size:11px;font-weight:800}.stTabs [aria-selected="true"]{color:#13ddff;border-bottom:2px solid #13ddff}
.stButton>button{border:1px solid #145074;background:linear-gradient(180deg,#092238,#061421);color:#dcecf6;border-radius:10px;font-weight:800;font-size:10px;letter-spacing:.3px}.stButton>button:hover{border-color:#1ce0ff;color:white;box-shadow:0 0 14px rgba(19,217,255,.12)}
button[kind="primary"]{background:linear-gradient(90deg,#0a8cff,#13d9ff)!important;border:0!important;color:#001018!important}
[data-testid="stDataFrame"]{border:1px solid #123b57;border-radius:12px;overflow:hidden}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    try: st.image("vs_pattern_lab_logo.png", use_container_width=True)
    except Exception: pass
    st.markdown('<div class="vs-brand"><div class="vs-brand-title">VS PATTERN <span>LAB</span></div><div class="vs-brand-sub">SCAN • STUDY • VALIDATE • RESEARCH</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section">RESEARCH CONTROLS</div>', unsafe_allow_html=True)
    market=st.selectbox("Market",["India","Global"])
    tf=st.selectbox("Timeframe",TFS,index=4)
    universe=st.radio("Universe",["Stocks","Indices","Custom"])
    if universe=="Stocks":
        base=INDIA_DEFAULT if market=="India" else GLOBAL_DEFAULT
        symbols=st.multiselect("Symbols",base,default=base[:10])
    elif universe=="Indices":
        names=st.multiselect("Indices",list(INDEX_SYMBOLS),default=list(INDEX_SYMBOLS)[:3]); symbols=[INDEX_SYMBOLS[x] for x in names]
    else:
        symbols=[x.strip() for x in st.text_area("Symbols", "RELIANCE,TCS,INFY" if market=="India" else "AAPL,MSFT,NVDA").split(",") if x.strip()]
    run=st.button("🔎  RUN RESEARCH SCAN",type="primary",use_container_width=True)
    st.markdown('<div class="vs-brand-sub" style="margin-top:10px">RESEARCH ONLY • NO EXECUTION</div>', unsafe_allow_html=True)

if "scan" not in st.session_state: st.session_state.scan=pd.DataFrame()
if run:
    with st.spinner("Running pattern + context engines..."):
        st.session_state.scan=scan_universe(symbols,tf,market)
res=st.session_state.scan

st.markdown(f"""<div class="hero"><div class="hero-top"><div><div class="hero-title">VS PATTERN <span>LAB</span> 📊</div><div class="hero-sub">Multi-Timeframe Market Pattern Intelligence • Research Engine • {market} • {tf}</div></div><div class="pill">VAIBHAV SHIRSAT · RESEARCH MODE</div></div></div>""", unsafe_allow_html=True)

pattern_count=len(res); bull_count=int((res.Direction=="Bullish").sum()) if not res.empty else 0; bear_count=int((res.Direction=="Bearish").sum()) if not res.empty else 0; confirmed_count=int(res.Status.isin(["Confirmed","Retest"]).sum()) if not res.empty else 0
regime="BULLISH" if bull_count>bear_count else ("BEARISH" if bear_count>bull_count else "NEUTRAL")
st.markdown('<div class="section-title">🌐 PATTERN RESEARCH COMMAND CENTER <small>decision-support research dashboard</small></div>', unsafe_allow_html=True)
c=st.columns(5)
with c[0]: st.markdown(f'<div class="regime"><div class="regime-tag">CURRENT REGIME</div><div class="regime-value">{regime}</div><div class="regime-note">Current scan distribution</div></div>',unsafe_allow_html=True)
with c[1]: st.markdown(f'<div class="card"><div class="card-label">PATTERN EVENTS</div><div class="card-value cyan">{pattern_count}</div><div class="card-note">Detected research events</div></div>',unsafe_allow_html=True)
with c[2]: st.markdown(f'<div class="card"><div class="card-label">BULLISH PATTERNS</div><div class="card-value green">{bull_count}</div><div class="card-note">Current timeframe</div></div>',unsafe_allow_html=True)
with c[3]: st.markdown(f'<div class="card"><div class="card-label">BEARISH PATTERNS</div><div class="card-value red">{bear_count}</div><div class="card-note">Current timeframe</div></div>',unsafe_allow_html=True)
with c[4]: st.markdown(f'<div class="card"><div class="card-label">CONFIRMED / RETEST</div><div class="card-value yellow">{confirmed_count}</div><div class="card-note">Lifecycle states</div></div>',unsafe_allow_html=True)

t1,t2,t3,t4,t5=st.tabs(["SCANNER","MTF RESEARCH","PATTERN BACKTEST","CHART","PATTERN UNIVERSE"])
with t1:
    st.markdown('<div class="section-title">🔎 Pattern Scanner <small>ranked by research score and lifecycle</small></div>',unsafe_allow_html=True)
    if res.empty: st.info("Run the research scan from the left panel.")
    else:
        f1,f2,f3=st.columns([1.2,1.8,1])
        with f1: min_score=st.slider("Minimum research score",0,100,60)
        with f2: status=st.multiselect("Lifecycle",["Formation","Developing","Confirmed","Retest","Failed","Invalidated"],default=["Confirmed","Retest","Developing"])
        with f3: cat=st.multiselect("Category",sorted(res.Category.unique()),default=sorted(res.Category.unique()))
        view=res[(res.Score>=min_score)&(res.Status.isin(status))&(res.Category.isin(cat))].sort_values(["Score","LifecycleScore","Symbol"],ascending=[False,False,True])
        st.dataframe(view,use_container_width=True,hide_index=True,height=430)
        st.caption("Research score is a ranking framework, not a probability or trade instruction.")
with t2:
    st.markdown('<div class="section-title">🧭 Multi-Timeframe Pattern Alignment <small>5M → 1MN weighted research</small></div>',unsafe_allow_html=True)
    sym=st.text_input("Research symbol",symbols[0] if symbols else ("RELIANCE" if market=="India" else "AAPL"),key="mtf_symbol")
    if st.button("RUN MTF RESEARCH",key="run_mtf"):
        mtf,summary=mtf_analysis(sym,market); a,b,c,d=st.columns(4);a.metric("Final Bias",summary["FinalBias"]);b.metric("Bull Weighted",summary["BullWeighted"]);c.metric("Bear Weighted",summary["BearWeighted"]);d.metric("Timeframes",len(mtf));st.dataframe(mtf,use_container_width=True,hide_index=True)
with t3:
    st.markdown('<div class="section-title">📚 Historical Pattern Research <small>point-in-time signals • next-open entry • no look-ahead</small></div>',unsafe_allow_html=True)
    bsym=st.text_input("Backtest symbol",symbols[0] if symbols else ("RELIANCE" if market=="India" else "AAPL"),key="bt_symbol"); btf=st.selectbox("Backtest timeframe",TFS,index=4,key="btf"); direction=st.selectbox("Research direction",["Bullish","Bearish"],key="btdir"); horizon=st.slider("Forward candles",1,50,10,key="horizon")
    if st.button("RUN BACKTEST",key="run_bt"):
        df=fetch_ohlcv(bsym,btf,market)
        if df.empty: st.error("No data returned.")
        else:
            trades,report=backtest_fixed_horizon(df,direction,horizon); wf=walk_forward(df,direction,horizon); cols=st.columns(5);cols[0].metric("Signals",report["signals"]);cols[1].metric("Win Rate",f"{report['win_rate_pct']:.1f}%");cols[2].metric("Avg Return",f"{report['avg_return_pct']:.2f}%");cols[3].metric("Profit Factor",f"{report['profit_factor']:.2f}");cols[4].metric("Max DD",f"{report['max_drawdown_pct']:.2f}%"); st.write("Out-of-sample validation"); st.json(wf); st.dataframe(trades.tail(100),use_container_width=True,hide_index=True)
with t4:
    st.markdown('<div class="section-title">📈 Pattern Chart <small>visual validation before interpretation</small></div>',unsafe_allow_html=True)
    csym=st.selectbox("Chart symbol",symbols if symbols else ["RELIANCE"],key="chart_symbol"); cdf=fetch_ohlcv(csym,tf,market)
    if not cdf.empty:
        fig=go.Figure(go.Candlestick(x=cdf.index,open=cdf.Open,high=cdf.High,low=cdf.Low,close=cdf.Close));fig.update_layout(height=650,xaxis_rangeslider_visible=False,title=f"{csym} • {tf} • OHLC Research Chart",paper_bgcolor="#06111b",plot_bgcolor="#06111b",font=dict(color="#dbeaf4"),margin=dict(l=10,r=10,t=55,b=10),xaxis=dict(gridcolor="#10283a"),yaxis=dict(gridcolor="#10283a"));st.plotly_chart(fig,use_container_width=True)
with t5:
    st.markdown('<div class="section-title">🧩 Pattern Universe <small>catalogue of the research engine</small></div>',unsafe_allow_html=True)
    groups={"Candlestick Patterns":"Hammer, Inverted Hammer, Hanging Man, Shooting Star, Doji, Dragonfly Doji, Gravestone Doji, Spinning Top, Marubozu, Pin Bar, Engulfing, Harami, Harami Cross, Piercing Line, Dark Cloud Cover, Tweezer, Kicker, Belt Hold, Counterattack, Separating Lines, Morning Star, Evening Star, Doji Stars, Three Soldiers, Three Crows, Three Inside/Outside, Abandoned Baby, Tri-Star, Tasuki Gap, Three Line Strike, Three Methods, Breakaway, Hikkake, Island Reversal","Chart / Geometry":"Head & Shoulders, Inverse H&S, Double Top/Bottom, Triple Top/Bottom, Ascending/Descending/Symmetrical Triangles, Rising/Falling Wedges, Flags, Pennants, Rectangle, Rounding, Cup & Handle, Diamond, Broadening, Bump & Run, Pipe, Scallops, Measured Move, V / Inverse V, HH/HL, LH/LL","Context / Research":"Trend, Support, Resistance, Trendline, Range, Breakout/Breakdown, Volume, Volatility, Momentum, Gap, Event studies"}
    cols=st.columns(3)
    for i,(k,v) in enumerate(groups.items()):
        with cols[i]: st.markdown(f'<div class="card" style="min-height:190px"><div class="card-label">{k.upper()}</div><div style="font-size:12px;line-height:1.75;margin-top:10px;color:#c7d9e5">{v}</div></div>',unsafe_allow_html=True)

st.divider();st.caption("VS PATTERN LAB is intentionally research-only. No broker/API integration, alerts, auto-trading, or live execution. Validate detected formations visually and with context before using research output.")
