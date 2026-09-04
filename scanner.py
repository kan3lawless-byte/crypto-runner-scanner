import os
import time
import math
import requests
import traceback

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "20"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "67"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS_PER_EXCHANGE", "60"))

# Prevent repeated alerts
last_alert = {}


# ============================================================
# TELEGRAM
# ============================================================

def telegram(message):
    print(message)

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=10,
        )

    except Exception as e:
        print("Telegram error:", e)


# ============================================================
# INDICATORS
# ============================================================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def macd(series):

    fast = ema(series, 12)
    slow = ema(series, 26)

    macd_line = fast - slow
    signal = ema(macd_line, 9)
    hist = macd_line - signal

    return macd_line, signal, hist


def atr(df, length=14):

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length).mean()


# ============================================================
# RUNNER ENGINE
# ============================================================

def analyze(df):

    if len(df) < 110:
        return None

    close = df["close"]
    volume = df["volume"]

    e9 = ema(close, 9)
    e21 = ema(close, 21)
    e25 = ema(close, 25)
    e50 = ema(close, 50)
    e99 = ema(close, 99)

    r = rsi(close)

    _, _, hist = macd(close)

    atr14 = atr(df)

    last = -1

    price = close.iloc[last]

    # -------------------------
    # Price acceleration
    # -------------------------

    pct_5 = (price / close.iloc[-2] - 1) * 100
    pct_15 = (price / close.iloc[-4] - 1) * 100
    pct_30 = (price / close.iloc[-7] - 1) * 100
    pct_60 = (price / close.iloc[-13] - 1) * 100

    direction = "LONG" if pct_15 >= 0 else "SHORT"

    sign = 1 if direction == "LONG" else -1

    score = 0
    reasons = []

    # ========================================================
    # 1. PRICE ACCELERATION - 30
    # ========================================================

    acceleration = sign * pct_15

    if acceleration >= 0.35:
        score += 8
        reasons.append("5/15m momentum active")

    if acceleration >= 0.75:
        score += 8
        reasons.append("price accelerating")

    if acceleration >= 1.25:
        score += 8
        reasons.append("strong price expansion")

    if sign * pct_30 > sign * pct_15:
        score += 3

    if sign * pct_60 > 0:
        score += 3


    # ========================================================
    # 2. VOLUME EXPANSION - 20
    # ========================================================

    vol_avg = volume.iloc[-21:-1].mean()

    volume_ratio = (
        volume.iloc[-1] / vol_avg
        if vol_avg and not math.isnan(vol_avg)
        else 0
    )

    if volume_ratio >= 1.3:
        score += 6
        reasons.append(f"volume {volume_ratio:.1f}x")

    if volume_ratio >= 1.8:
        score += 7

    if volume_ratio >= 2.5:
        score += 7
        reasons.append("abnormal volume")


    # ========================================================
    # 3. EMA TREND / RIBBON - 15
    # ========================================================

    bullish_stack = (
        e9.iloc[-1] > e21.iloc[-1] > e25.iloc[-1] > e50.iloc[-1]
    )

    bearish_stack = (
        e9.iloc[-1] < e21.iloc[-1] < e25.iloc[-1] < e50.iloc[-1]
    )

    if direction == "LONG" and bullish_stack:
        score += 10
        reasons.append("EMA ribbon bullish")

    if direction == "SHORT" and bearish_stack:
        score += 10
        reasons.append("EMA ribbon bearish")

    ema_gap_now = abs(e9.iloc[-1] - e21.iloc[-1])
    ema_gap_old = abs(e9.iloc[-3] - e21.iloc[-3])

    if ema_gap_now > ema_gap_old:
        score += 5
        reasons.append("EMA ribbon expanding")


    # ========================================================
    # 4. MACD ACCELERATION - 15
    # ========================================================

    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]

    macd_direction = (
        hist_now > 0 if direction == "LONG" else hist_now < 0
    )

    macd_accel = (
        hist_now > hist_prev
        if direction == "LONG"
        else hist_now < hist_prev
    )

    if macd_direction:
        score += 7

    if macd_accel:
        score += 8
        reasons.append("MACD accelerating")


    # ========================================================
    # 5. RSI MOMENTUM - 10
    # ========================================================

    current_rsi = r.iloc[-1]
    previous_rsi = r.iloc[-2]

    if direction == "LONG":

        if current_rsi >= 55:
            score += 5

        if current_rsi > previous_rsi:
            score += 5
            reasons.append(f"RSI {current_rsi:.0f} rising")

    else:

        if current_rsi <= 45:
            score += 5

        if current_rsi < previous_rsi:
            score += 5
            reasons.append(f"RSI {current_rsi:.0f} falling")


    # ========================================================
    # 6. RANGE / ATR EXPANSION - 10
    # ========================================================

    candle_range = df["high"].iloc[-1] - df["low"].iloc[-1]

    atr_now = atr14.iloc[-1]

    atr_ratio = (
        candle_range / atr_now
        if atr_now and not math.isnan(atr_now)
        else 0
    )

    if atr_ratio >= 1.0:
        score += 5

    if atr_ratio >= 1.5:
        score += 5
        reasons.append("range expanding")


    # ========================================================
    # FADING DETECTION
    # ========================================================

    fading = False

    if direction == "LONG":

        if (
            hist_now < hist_prev
            and volume_ratio < 1.0
            and e9.iloc[-1] <= e9.iloc[-2]
        ):
            fading = True

    else:

        if (
            hist_now > hist_prev
            and volume_ratio < 1.0
            and e9.iloc[-1] >= e9.iloc[-2]
        ):
            fading = True


    # ========================================================
    # STATE
    # ========================================================

    if fading:
        state = "FADING"

    elif score >= 88:
        state = "STRONG RUNNING"

    elif score >= 75:
        state = "RUNNING"

    elif score >= MIN_SCORE:
        state = "IGNITION"

    else:
        return None


    return {
        "direction": direction,
        "state": state,
        "score": min(score, 100),

        "price": price,

        "pct5": pct_5,
        "pct15": pct_15,
        "pct30": pct_30,
        "pct60": pct_60,

        "volume_ratio": volume_ratio,
        "rsi": current_rsi,

        "reasons": reasons,
    }


# ============================================================
# EXCHANGES
# ============================================================

def make_exchanges():

    exchanges = []

    # Kraken SPOT
    exchanges.append(
        (
            "KRAKEN",
            ccxt.kraken({
                "enableRateLimit": True,
            }),
            "spot",
        )
    )

    # Coinbase SPOT
    exchanges.append(
        (
            "COINBASE",
            ccxt.coinbase({
                "enableRateLimit": True,
            }),
            "spot",
        )
    )

    # Bybit USDT PERPETUALS
    exchanges.append(
        (
            "BYBIT-PERP",
            ccxt.bybit({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap"
                },
            }),
            "swap",
        )
    )

    # OKX PERPETUALS
    exchanges.append(
        (
            "OKX-PERP",
            ccxt.okx({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap"
                },
            }),
            "swap",
        )
    )

    return exchanges


# ============================================================
# SYMBOL DISCOVERY
# ============================================================

def get_symbols(exchange, market_type):

    exchange.load_markets()

    result = []

    for symbol, market in exchange.markets.items():

        if not market.get("active", True):
            continue

        if market_type == "spot":

            if not market.get("spot"):
                continue

            quote = market.get("quote")

            if quote not in ["USD", "USDT"]:
                continue

        else:

            if not market.get("swap"):
                continue

            if not market.get("linear"):
                continue

            quote = market.get("quote")

            if quote not in ["USDT", "USDC"]:
                continue

        result.append(symbol)

    return result[:MAX_SYMBOLS]


# ============================================================
# ALERT CONTROL
# ============================================================

def should_alert(key, signal):

    previous = last_alert.get(key)

    if previous is None:
        return True

    previous_state = previous["state"]
    previous_score = previous["score"]

    if signal["state"] != previous_state:
        return True

    if signal["score"] >= previous_score + 7:
        return True

    return False


# ============================================================
# EXCHANGE SCAN
# ============================================================

def scan_exchange(name, exchange, market_type):

    try:
        symbols = get_symbols(exchange, market_type)

    except Exception as e:

        print(f"{name} market loading failed:", e)
        return


    signals = []

    for symbol in symbols:

        try:

            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe="5m",
                limit=130,
            )

            if not candles:
                continue

            df = pd.DataFrame(
                candles,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )

            signal = analyze(df)

            if signal:

                signal["symbol"] = symbol
                signal["exchange"] = name

                signals.append(signal)

        except Exception:
            continue


    signals.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Only top runners per cycle
    for signal in signals[:5]:

        key = (
            signal["exchange"],
            signal["symbol"],
            signal["direction"],
        )

        if not should_alert(key, signal):
            continue

        reasons = ", ".join(signal["reasons"][:5])

        msg = (
            f"🚨 {signal['exchange']} RUNNER\n\n"
            f"{signal['symbol']} — {signal['direction']}\n"
            f"{signal['state']} — {signal['score']}/100\n\n"
            f"5m: {signal['pct5']:+.2f}%\n"
            f"15m: {signal['pct15']:+.2f}%\n"
            f"30m: {signal['pct30']:+.2f}%\n"
            f"1h: {signal['pct60']:+.2f}%\n\n"
            f"Volume: {signal['volume_ratio']:.2f}x normal\n"
            f"RSI: {signal['rsi']:.1f}\n\n"
            f"{reasons}\n\n"
            f"Verify the same direction on KCEX/BTCC before entering."
        )

        telegram(msg)

        last_alert[key] = {
            "state": signal["state"],
            "score": signal["score"],
            "time": time.time(),
        }


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    exchanges = make_exchanges()

    telegram(
        "🟢 Crypto Runner Scanner started.\n"
        "Scanning Kraken, Coinbase, Bybit and OKX."
    )

    while True:

        start = time.time()

        for name, exchange, market_type in exchanges:

            try:

                print(f"\nScanning {name}...")

                scan_exchange(
                    name,
                    exchange,
                    market_type,
                )

            except Exception as e:

                print(
                    f"{name} failed:",
                    e,
                )

                traceback.print_exc()

        elapsed = time.time() - start

        sleep_for = max(
            1,
            SCAN_SECONDS - elapsed,
        )

        print(
            f"\nCycle complete. "
            f"Next scan in {sleep_for:.0f}s."
        )

        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
