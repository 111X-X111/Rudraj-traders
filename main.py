#!/usr/bin/env python3
# main_soft_refresh_alerts.py — Streamlit OI tracker with soft auto-refresh, countdown timer, and visual alerts
#
# Install:
#   pip install streamlit streamlit-autorefresh kiteconnect pandas pytz
#
# Run:
#   streamlit run main_soft_refresh_alerts.py

import os
import json
import time
import threading
from datetime import datetime, date, timedelta, timezone, time as dtime

import pandas as pd
import pytz
import streamlit as st
from kiteconnect import KiteConnect
import math

# Try to import streamlit-autorefresh (soft rerun). If not available, we'll fallback.
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    HAVE_STREAMLIT_AUTOREFRESH = True
except Exception:
    HAVE_STREAMLIT_AUTOREFRESH = False

# ---------- Config ----------
HISTORY_FILE = "oi_history.json"
HISTORICAL_MAX_MINUTES = 120
BOOTSTRAP_MINUTES = 60        # minutes fetched on startup to bootstrap session history
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30)
PCT_CHANGE_THRESHOLDS = {5: 8.0, 10: 10.0, 15: 15.0, 30: 25.0}
IST = pytz.timezone("Asia/Kolkata")

EXCHANGE_NFO = "NFO"
EXCHANGE_LTP = "NSE"

# NOTE: keys copied from your original file; keep secure.
API_KEY = "afzia78cwraaod5x"
API_SECRET = "b527807j5ilcndjp5u2jhu9znrjxz35e"

# ---------- Utilities: persistence ----------
def _dt_to_iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def _iso_to_dt(s):
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def load_persisted_history(path=HISTORY_FILE):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out = {}
    for key, arr in raw.items():
        lst = []
        for item in arr:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            ts = _iso_to_dt(item[0])
            oi = item[1]
            if oi is not None:
                try:
                    oi = int(oi)
                except Exception:
                    pass
            lst.append((ts, oi))
        if lst:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=HISTORICAL_MAX_MINUTES)
            lst = [t for t in lst if t[0] is not None and t[0] >= cutoff]
        out[key] = lst
    return out

def save_persisted_history(history: dict, path=HISTORY_FILE):
    tmp = path + ".tmp"
    raw = {}
    for key, lst in history.items():
        arr = []
        for t, oi in lst:
            arr.append([_dt_to_iso(t), oi])
        raw[key] = arr
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass

# ---------- Helpers ----------
def check_password():
    def password_entered():
        if st.session_state["password"] == "Rudraj@911":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect ❌")
        return False
    else:
        return True

def get_atm_strike(kite_obj: KiteConnect, underlying_sym: str, exch_for_ltp: str, strike_diff: int):
    try:
        ltp_inst = f"{exch_for_ltp}:{underlying_sym}"
        ltp_data = kite_obj.ltp(ltp_inst)
        if not ltp_data or ltp_inst not in ltp_data or "last_price" not in ltp_data[ltp_inst]:
            return None, None
        ltp = ltp_data[ltp_inst]["last_price"]
        atm = round(ltp / strike_diff) * strike_diff
        return atm, ltp
    except Exception:
        return None, None

def list_expiries(instruments: list, underlying_prefix: str):
    today = date.today()
    expiries = set()
    for inst in instruments:
        if inst.get("name") == underlying_prefix and inst.get("exchange") == EXCHANGE_NFO:
            ex = inst.get("expiry")
            if isinstance(ex, date) and ex >= today:
                expiries.add(ex)
    return sorted(list(expiries))

def symbol_prefix_for_expiry(instruments: list, underlying_prefix: str, expiry_dt: date):
    for inst in instruments:
        if (inst.get("name") == underlying_prefix and inst.get("exchange") == EXCHANGE_NFO and inst.get("expiry") == expiry_dt):
            ts = inst.get("tradingsymbol", "")
            return ts[:len(underlying_prefix)+5] if ts else ""
    return ""

def safe_fmt(val, fmt):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        return fmt.format(val)
    except Exception:
        return str(val) if val is not None else ""

# ---------- Option details finder ----------
def get_relevant_option_details(instruments: list, atm: float, expiry_dt: date,
                                strike_diff: int, levels: int,
                                underlying_prefix: str, symbol_prefix: str):
    out = {}
    if not expiry_dt or atm is None:
        return out
    for i in range(-levels, levels+1):
        strike = int(atm + i * strike_diff)
        ce_found = None
        pe_found = None
        for inst in instruments:
            if (inst.get("name") == underlying_prefix and
                inst.get("strike") == strike and
                inst.get("expiry") == expiry_dt and
                inst.get("exchange") == EXCHANGE_NFO):
                ts = inst.get("tradingsymbol", "")
                if inst.get("instrument_type") == "CE" and symbol_prefix in ts:
                    ce_found = inst
                if inst.get("instrument_type") == "PE" and symbol_prefix in ts:
                    pe_found = inst
            if ce_found and pe_found:
                break
        suffix = "atm" if i == 0 else (f"itm{-i}" if i < 0 else f"otm{i}")
        if ce_found:
            out[f"{suffix}_ce"] = {
                "tradingsymbol": ce_found.get("tradingsymbol"),
                "instrument_token": ce_found.get("instrument_token"),
                "strike": strike
            }
        if pe_found:
            out[f"{suffix}_pe"] = {
                "tradingsymbol": pe_found.get("tradingsymbol"),
                "instrument_token": pe_found.get("instrument_token"),
                "strike": strike
            }
    return out

# ---------- Market time helpers ----------
def is_market_closed_now():
    ist_now = datetime.now(IST)
    market_close_ist = dtime(hour=15, minute=30)
    return ist_now.time() >= market_close_ist

def market_close_datetime_utc(ref_date: date):
    close_local = datetime.combine(ref_date, dtime(hour=15, minute=30))
    close_localized = IST.localize(close_local)
    return close_localized.astimezone(timezone.utc)

# ---------- History & live handling with persistence ----------
def _ensure_session_struct():
    if "oi_history" not in st.session_state:
        st.session_state["oi_history"] = load_persisted_history()
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = 0.0
    if "last_quote_success" not in st.session_state:
        st.session_state["last_quote_success"] = None
    if "refresh_count" not in st.session_state:
        st.session_state["refresh_count"] = 0
    if "bootstrap_done_for" not in st.session_state:
        st.session_state["bootstrap_done_for"] = set()
    if "offline" not in st.session_state:
        st.session_state["offline"] = False
    # timestamp of when UI last finished processing (used for countdown)
    if "ui_last_processed_ts" not in st.session_state:
        st.session_state["ui_last_processed_ts"] = time.time()

def persist_history_after_change():
    try:
        now_utc = datetime.now(timezone.utc)
        pruned = {}
        for k, lst in st.session_state["oi_history"].items():
            if not lst:
                continue
            cutoff = now_utc - timedelta(minutes=HISTORICAL_MAX_MINUTES)
            newlst = [ (t,o) for (t,o) in lst if t is not None and t >= cutoff ]
            pruned[k] = newlst
        save_persisted_history(pruned)
    except Exception:
        pass

def update_live_oi_from_quote(kite: KiteConnect, details: dict):
    _ensure_session_struct()
    now_utc = datetime.now(timezone.utc)

    prefix = "NFO:"
    sym_map = {}
    quoted_symbols = []
    for k, d in details.items():
        ts = d.get("tradingsymbol")
        if not ts:
            continue
        qk = f"{prefix}{ts}"
        sym_map[qk] = k
        quoted_symbols.append(qk)
    if not quoted_symbols:
        return

    try:
        quotes = kite.quote(quoted_symbols)
        st.session_state["offline"] = False
    except Exception:
        st.session_state["offline"] = True
        return

    any_success = False
    for qk, key in sym_map.items():
        entry = quotes.get(qk) or quotes.get(qk.replace("NFO:", "")) or {}
        oi = None
        if isinstance(entry, dict):
            oi = entry.get("oi")
        if oi is not None:
            try:
                oi = int(oi)
            except Exception:
                pass

        hist = st.session_state["oi_history"].setdefault(key, [])
        hist.append((now_utc, oi))
        cutoff = now_utc - timedelta(minutes=max(HISTORICAL_MAX_MINUTES, max(OI_CHANGE_INTERVALS_MIN)))
        st.session_state["oi_history"][key] = [t for t in hist if t[0] is not None and t[0] >= cutoff]

        if oi is not None:
            any_success = True

    if any_success:
        st.session_state["last_quote_success"] = now_utc
    st.session_state["refresh_count"] += 1
    persist_history_after_change()

def fetch_last_oi_at_close(kite: KiteConnect, details: dict):
    _ensure_session_struct()
    today_ist = datetime.now(IST).date()
    close_utc = market_close_datetime_utc(today_ist)
    from_utc = close_utc - timedelta(minutes=max(HISTORICAL_MAX_MINUTES, max(OI_CHANGE_INTERVALS_MIN)))

    try:
        _ = kite.profile()
        st.session_state["offline"] = False
    except Exception:
        st.session_state["offline"] = True
        return

    for key, d in details.items():
        tok = d.get("instrument_token")
        if not tok:
            continue
        existing = st.session_state["oi_history"].get(key)
        if existing:
            last_ts = existing[-1][0] if existing else None
            if last_ts and last_ts >= close_utc:
                continue
        try:
            candles = kite.historical_data(
                instrument_token=tok,
                from_date=from_utc,
                to_date=close_utc + timedelta(seconds=1),
                interval="minute",
                oi=True,
                continuous=False
            )
            if not candles:
                continue
            chosen = None
            for c in reversed(candles):
                ct = c.get("date")
                if ct and ct <= close_utc and c.get("oi") is not None:
                    ts_utc = ct.astimezone(timezone.utc)
                    chosen = (ts_utc, int(c.get("oi")))
                    break
            if chosen:
                st.session_state["oi_history"].setdefault(key, []).append(chosen)
                st.session_state["last_quote_success"] = chosen[0]
        except Exception:
            continue
    persist_history_after_change()

def bootstrap_history_from_historical_minutes(kite: KiteConnect, details: dict, minutes: int = BOOTSTRAP_MINUTES):
    _ensure_session_struct()
    now_utc = datetime.now(timezone.utc)
    to_dt = now_utc
    from_dt = to_dt - timedelta(minutes=minutes)

    try:
        _ = kite.profile()
        st.session_state["offline"] = False
    except Exception:
        st.session_state["offline"] = True
        return

    for key, d in details.items():
        tok = d.get("instrument_token")
        if not tok:
            st.session_state["bootstrap_done_for"].add(key)
            continue
        if key in st.session_state["bootstrap_done_for"]:
            continue
        try:
            candles = kite.historical_data(
                instrument_token=tok,
                from_date=from_dt,
                to_date=to_dt,
                interval="minute",
                oi=True,
                continuous=False
            )
            if not candles:
                st.session_state["bootstrap_done_for"].add(key)
                continue
            hist_list = []
            for c in candles:
                ct = c.get("date")
                oi = c.get("oi")
                if ct:
                    ct_utc = ct.astimezone(timezone.utc)
                else:
                    ct_utc = None
                if oi is not None:
                    try:
                        oi = int(oi)
                    except Exception:
                        pass
                hist_list.append((ct_utc, oi))
            existing = st.session_state["oi_history"].get(key, [])
            combined = existing + hist_list
            combined_map = {}
            for t,o in combined:
                if t is None:
                    continue
                combined_map[t.isoformat()] = (t, o)
            combined_sorted = [combined_map[k] for k in sorted(combined_map.keys())]
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=HISTORICAL_MAX_MINUTES)
            combined_sorted = [t for t in combined_sorted if t[0] >= cutoff]
            st.session_state["oi_history"][key] = combined_sorted
            st.session_state["bootstrap_done_for"].add(key)
            if combined_sorted:
                st.session_state["last_quote_success"] = combined_sorted[-1][0]
        except Exception:
            st.session_state["bootstrap_done_for"].add(key)
            continue
    persist_history_after_change()

def compute_pct_changes_from_history(details: dict, intervals=(5,10,15,30)):
    _ensure_session_struct()
    out = {}
    now_utc = datetime.now(timezone.utc)
    hist_all = st.session_state.get("oi_history", {})

    for key, d in details.items():
        hist = hist_all.get(key, [])
        hist = sorted(hist, key=lambda x: x[0]) if hist else hist
        latest_oi = None
        latest_ts = None
        if hist:
            for t, o in reversed(hist):
                if o is not None:
                    latest_ts = t
                    latest_oi = o
                    break
            if latest_ts is None and hist:
                latest_ts = hist[-1][0]
                latest_oi = None
        out[key] = {"latest_oi": latest_oi, "latest_oi_timestamp": latest_ts}
        for m in intervals:
            past_target = now_utc - timedelta(minutes=m)
            past_oi = None
            used_earliest = False
            for t, o in reversed(hist):
                if t <= past_target and o is not None:
                    past_oi = o
                    break
            if past_oi is None:
                for t, o in hist:
                    if o is not None:
                        past_oi = o
                        used_earliest = True
                        break
            pct = None
            if past_oi not in (None, 0) and latest_oi not in (None,):
                try:
                    pct = (latest_oi - past_oi) / past_oi * 100.0
                except Exception:
                    pct = None
            out[key][f"pct_diff_{m}m"] = pct
            out[key][f"pct_used_earliest_{m}m"] = used_earliest
    return out

def build_table(report: dict, details: dict, atm: float, step: int, levels: int, intervals=(5,10,15,30), is_call=True):
    rows = []
    for i in range(-levels, levels+1):
        strike = atm + i*step
        suffix = "atm" if i==0 else ("itm{}".format(-i) if i<0 else "otm{}".format(i))
        key = f"{suffix}_{'ce' if is_call else 'pe'}"
        d = details.get(key, {})
        r = report.get(key, {})
        latest_ts = r.get("latest_oi_timestamp")
        ist_time_str = latest_ts.astimezone(IST).strftime("%H:%M:%S") if latest_ts else None
        row = {
            "Strike": int(d.get("strike", strike)),
            "Symbol": d.get("tradingsymbol", "N/A"),
            "Latest OI": r.get("latest_oi"),
            "OI Time (IST)": ist_time_str
        }
        for m in intervals:
            row[f"OI %Chg ({m}m)"] = r.get(f"pct_diff_{m}m")
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- Background worker ----------
_BG_LOCK = threading.Lock()
_BG_STARTED_FLAG = "_oi_tracker_bg_started"

def _background_worker_loop(default_underlying="NIFTY 50", default_levels=2, strike_diff_map=None, interval_sec=60):
    if strike_diff_map is None:
        strike_diff_map = {"NIFTY 50": 50, "NIFTY BANK": 100}

    def merge_and_save(persist_path, new_data):
        persisted = load_persisted_history(persist_path)
        now_utc = datetime.now(timezone.utc)
        for k, lst in new_data.items():
            existing = persisted.get(k, [])
            combined = existing + lst
            cmap = {}
            for t,o in combined:
                if t is None:
                    continue
                cmap[t.isoformat()] = (t,o)
            combined_sorted = [cmap[k] for k in sorted(cmap.keys())]
            cutoff = now_utc - timedelta(minutes=HISTORICAL_MAX_MINUTES)
            combined_sorted = [t for t in combined_sorted if t[0] >= cutoff]
            persisted[k] = combined_sorted
        save_persisted_history(persisted, persist_path)

    while True:
        try:
            access_token = None
            if os.path.exists("access_token.txt"):
                try:
                    with open("access_token.txt", "r") as f:
                        access_token = f.read().strip()
                except Exception:
                    access_token = None

            if not access_token:
                time.sleep(interval_sec)
                continue

            kite = KiteConnect(api_key=API_KEY)
            try:
                kite.set_access_token(access_token)
                _ = kite.profile()
            except Exception:
                time.sleep(interval_sec)
                continue

            try:
                instruments = kite.instruments(EXCHANGE_NFO)
            except Exception:
                instruments = []

            underlying = default_underlying
            strike_diff = strike_diff_map.get(underlying, 50)
            levels = default_levels

            try:
                atm, ltp = get_atm_strike(kite, underlying, EXCHANGE_LTP, strike_diff)
            except Exception:
                atm, ltp = None, None

            if atm is None or not instruments:
                time.sleep(interval_sec)
                continue

            expiries = list_expiries(instruments, underlying.split(" ")[0].upper())
            if not expiries:
                time.sleep(interval_sec)
                continue
            target_expiry = expiries[0]
            symbol_prefix = symbol_prefix_for_expiry(instruments, underlying.split(" ")[0].upper(), target_expiry)

            details = get_relevant_option_details(instruments, atm, target_expiry, strike_diff, levels, underlying.split(" ")[0].upper(), symbol_prefix)
            if not details:
                time.sleep(interval_sec)
                continue

            new_hist_data = {}
            now_utc = datetime.now(timezone.utc)
            from_dt = now_utc - timedelta(minutes=BOOTSTRAP_MINUTES)
            to_dt = now_utc
            for key, d in details.items():
                tok = d.get("instrument_token")
                if not tok:
                    continue
                try:
                    candles = kite.historical_data(
                        instrument_token=tok,
                        from_date=from_dt,
                        to_date=to_dt,
                        interval="minute",
                        oi=True,
                        continuous=False
                    )
                except Exception:
                    candles = None
                hist_list = []
                if candles:
                    for c in candles:
                        ct = c.get("date")
                        oi = c.get("oi")
                        if ct:
                            ct_utc = ct.astimezone(timezone.utc)
                        else:
                            ct_utc = None
                        if oi is not None:
                            try:
                                oi = int(oi)
                            except Exception:
                                pass
                        hist_list.append((ct_utc, oi))
                if hist_list:
                    new_hist_data[key] = hist_list

            if new_hist_data:
                try:
                    merge_and_save(HISTORY_FILE, new_hist_data)
                except Exception:
                    pass

            quote_symbols = []
            prefix = "NFO:"
            sym_map = {}
            for k,d in details.items():
                ts = d.get("tradingsymbol")
                if not ts:
                    continue
                qk = f"{prefix}{ts}"
                quote_symbols.append(qk)
                sym_map[qk] = k
            try:
                quotes = kite.quote(quote_symbols)
            except Exception:
                quotes = {}

            now_utc = datetime.now(timezone.utc)
            new_quote_hist = {}
            for qk, key in sym_map.items():
                entry = quotes.get(qk) or quotes.get(qk.replace("NFO:", "")) or {}
                oi = None
                if isinstance(entry, dict):
                    oi = entry.get("oi")
                if oi is not None:
                    try:
                        oi = int(oi)
                    except Exception:
                        pass
                new_quote_hist.setdefault(key, []).append((now_utc, oi))

            if new_quote_hist:
                try:
                    merge_and_save(HISTORY_FILE, new_quote_hist)
                except Exception:
                    pass

        except Exception:
            pass

        time.sleep(interval_sec)

def start_background_worker_if_needed(interval_sec=60, default_underlying="NIFTY 50", default_levels=2):
    with _BG_LOCK:
        if st.session_state.get(_BG_STARTED_FLAG):
            return
        t = threading.Thread(
            target=_background_worker_loop,
            kwargs={"default_underlying": default_underlying, "default_levels": default_levels, "interval_sec": interval_sec},
            daemon=True,
            name="oi-tracker-bg"
        )
        t.start()
        st.session_state[_BG_STARTED_FLAG] = True

# ---------- Streamlit UI ----------
st.set_page_config(page_title="OI Tracker (Soft Auto-refresh + Alerts)", layout="wide")

if not check_password():
    st.stop()

st.title("📊 OI Tracker — Live OI + local persistence (soft auto-refresh + alerts)")

kite = KiteConnect(api_key=API_KEY)

with st.sidebar:
    if "access_token" not in st.session_state:
        st.title("Zerodha Kite Access Token Generator")
        login_url = kite.login_url()
        st.write("### Step 1: Login using this URL")
        st.markdown(f"[Click here to login]({login_url})")
        request_token = st.text_input("Paste the request token here:")
        if st.button("Get Access Token"):
            if request_token:
                try:
                    data = kite.generate_session(request_token, api_secret=API_SECRET)
                    st.session_state["access_token"] = data["access_token"]
                    with open("access_token.txt", "w") as f:
                        f.write(st.session_state["access_token"])
                    st.success("Access token stored in session and access_token.txt")
                except Exception as e:
                    st.error(f"Error generating session: {e}")
            else:
                st.warning("Paste request token first.")
    else:
        st.success("🔐 Access Token in session.")

    st.header("Market & Filters")
    underlying = st.selectbox("Underlying", ("NIFTY 50", "NIFTY BANK"), index=0)
    strike_diff = 50 if underlying == "NIFTY 50" else 100
    levels = st.slider("Strikes each side of ATM", 1, 6, 2)

    # Soft UI refresh control
    refresh = st.number_input("Refresh seconds (UI soft refresh)", min_value=5, max_value=600, value=30, step=1)
    autorun = st.checkbox("Auto-refresh (UI soft rerun)", value=True)
    go = st.button("Run / Refresh")

    st.markdown("---")
    st.subheader("Background fetch (server-side)")
    bg_interval = st.number_input("Background interval (seconds)", min_value=15, max_value=3600, value=60, step=15)
    bg_enable = st.checkbox("Enable background fetch thread (keeps HISTORY_FILE updated)", value=True)
    st.caption("Background worker requires access_token.txt in the app folder (or generate above). It runs as long as the Streamlit process is alive on the server.")

market_closed = is_market_closed_now()

# Auto-refresh (soft) for UI only
if autorun and not market_closed:
    if HAVE_STREAMLIT_AUTOREFRESH:
        st_autorefresh(interval=refresh * 1000, key="oi_soft_refresh")
        st.caption(f"🔄 Soft auto-refresh ON — every {refresh} seconds")
    else:
        last = st.session_state.get("last_refresh", 0.0)
        now_ts = time.time()
        if now_ts - last > refresh:
            st.session_state["last_refresh"] = now_ts
            try:
                st.experimental_rerun()
            except Exception:
                pass

# ensure session history loaded and ui timestamp exists
_ensure_session_struct()

# start background worker if requested
if bg_enable:
    start_background_worker_if_needed(interval_sec=int(bg_interval), default_underlying=underlying, default_levels=levels)

if "access_token" not in st.session_state:
    st.info("Generate Access Token in the sidebar to begin.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(st.session_state["access_token"])

try:
    prof = kite.profile()
    st.success(f"Connected: {prof.get('user_id')} — {prof.get('user_name')}")
    st.session_state["offline"] = False
except Exception:
    st.warning("Could not verify Kite login — working in OFFLINE mode using persisted history if available.")
    st.session_state["offline"] = True

@st.cache_data(ttl=600)
def load_instruments(nfo_exchange: str):
    return kite.instruments(nfo_exchange)

try:
    instruments = load_instruments(EXCHANGE_NFO)
except Exception:
    instruments = []
    st.warning("Failed to load instruments (offline or API error). Some features may not work.")

under_prefix = underlying.split(" ")[0].upper()
expiries = list_expiries(instruments, under_prefix) if instruments else []
if not expiries:
    st.error("No future expiries found (instruments unavailable).")
    st.stop()

exp_strs = [ex.strftime("%d-%b-%Y (%a)") for ex in expiries]
sel_idx = st.sidebar.selectbox("Target Expiry", range(len(expiries)), format_func=lambda i: exp_strs[i], index=0)
target_expiry = expiries[sel_idx]
symbol_prefix = symbol_prefix_for_expiry(instruments, under_prefix, target_expiry)

# ---------- UI countdown (approx) ----------
# compute seconds since last UI processing finish and show time-left until next soft refresh
now_ts = time.time()
last_proc = st.session_state.get("ui_last_processed_ts", now_ts)
elapsed_since_last_proc = now_ts - last_proc
time_left = max(0, int(refresh - elapsed_since_last_proc))
st.metric("Next soft refresh (approx sec)", f"{time_left}s")

if go or autorun or market_closed:
    atm, ltp = get_atm_strike(kite, underlying, EXCHANGE_LTP, strike_diff)
    if atm is None:
        st.error("Unable to compute ATM. Check connectivity or API keys.")
        st.stop()

    st.subheader(f"{underlying} — Spot: {ltp:.2f} | ATM: {int(atm)} | Expiry: {target_expiry.strftime('%d-%b-%Y')}")
    last_q = st.session_state.get("last_quote_success")
    if last_q:
        st.caption(f"Last successful quote (IST): {last_q.astimezone(IST).strftime('%d-%b-%Y %H:%M:%S')}")
    else:
        st.caption("No successful quote yet this session (using persisted data if available).")

    if st.session_state.get("offline"):
        st.info("OFFLINE: showing persisted/bootstrapped history (no live polling).")
    else:
        if market_closed:
            st.info("Market closed — showing last available values (no polling).")
        else:
            if autorun:
                st.caption(f"Auto-refresh ON — interval {refresh}s.")
            else:
                st.caption("Auto-refresh OFF — click Run / Refresh to fetch once.")

    details = get_relevant_option_details(instruments, atm, target_expiry, strike_diff, levels, under_prefix, symbol_prefix)
    if not details:
        st.warning("No matching option contracts found around ATM for selected expiry.")
        st.stop()

    if not market_closed and not st.session_state.get("offline"):
        need_bootstrap = False
        now_utc = datetime.now(timezone.utc)
        for key in details.keys():
            hist = st.session_state.get("oi_history", {}).get(key, [])
            if not hist:
                need_bootstrap = True
                break
            last_ts = hist[-1][0] if hist else None
            if not last_ts or last_ts < (now_utc - timedelta(minutes=BOOTSTRAP_MINUTES - 2)):
                need_bootstrap = True
                break
        if need_bootstrap:
            bootstrap_history_from_historical_minutes(kite, details, minutes=BOOTSTRAP_MINUTES)

    if not market_closed and not st.session_state.get("offline"):
        update_live_oi_from_quote(kite, details)
    else:
        need_fetch = False
        for key in details.keys():
            if not st.session_state["oi_history"].get(key):
                need_fetch = True
                break
        if need_fetch and not st.session_state.get("offline"):
            fetch_last_oi_at_close(kite, details)

    report = compute_pct_changes_from_history(details, intervals=OI_CHANGE_INTERVALS_MIN)

    calls_df = build_table(report, details, atm, strike_diff, levels, is_call=True)
    puts_df  = build_table(report, details, atm, strike_diff, levels, is_call=False)

    # ---------- Visual alerts styling ----------
    # create a styler that highlights OI %Chg columns that breach thresholds
    def style_breaches(df):
        df_sty = df.style
        # highlight cells per-value
        def highlight_cell(val, thr):
            try:
                if val is None:
                    return ""
                v = float(val)
            except Exception:
                return ""
            if abs(v) >= thr:
                return "background-color: #ffcccc; color: #700000; font-weight:700"  # red-ish
            if abs(v) >= 0.8 * thr:
                return "background-color: #fff5cc; color: #7a5200"  # yellow-ish
            return ""
        for m in OI_CHANGE_INTERVALS_MIN:
            col = f"OI %Chg ({m}m)"
            if col in df.columns:
                thr = PCT_CHANGE_THRESHOLDS.get(m, 10.0)
                df_sty = df_sty.applymap(lambda v, thr=thr: highlight_cell(v, thr), subset=pd.IndexSlice[:, [col]])
        return df_sty

    fmt_cols = {"Latest OI": lambda v: safe_fmt(v, "{:,.0f}") if v not in (None, "") else ""}
    for m in OI_CHANGE_INTERVALS_MIN:
        col = f"OI %Chg ({m}m)"
        fmt_cols[col] = (lambda f: (lambda v: safe_fmt(v, f)))("{:+.2f}%")

    st.markdown("### Calls (ΔOI %)")
    try:
        st.dataframe(style_breaches(calls_df).format(fmt_cols), use_container_width=True)
    except Exception:
        # fallback if styling fails
        st.dataframe(calls_df.style.format(fmt_cols), use_container_width=True)

    st.markdown("### Puts (ΔOI %)")
    try:
        st.dataframe(style_breaches(puts_df).format(fmt_cols), use_container_width=True)
    except Exception:
        st.dataframe(puts_df.style.format(fmt_cols), use_container_width=True)

    # Debug + persistence info
    st.markdown("### Debug: history preview (symbol → last_ts IST, last_oi, history_len, recent entries, bootstrapped?, persisted?)")
    dbg_rows = []
    persisted_exists = os.path.exists(HISTORY_FILE)
    for key, d in details.items():
        hist = st.session_state["oi_history"].get(key, [])
        last_ts = hist[-1][0].astimezone(IST).strftime("%H:%M:%S") if hist else ""
        last_oi = hist[-1][1] if hist else None
        hist_len = len(hist)
        recent = []
        for t, o in hist[-6:]:
            recent.append(f"{t.astimezone(IST).strftime('%H:%M:%S')}:{(o if o is not None else 'None')}")
        bootstrapped = "yes" if key in st.session_state.get("bootstrap_done_for", set()) else "-"
        dbg_rows.append({"Symbol": d.get("tradingsymbol", ""), "Last TS(IST)": last_ts, "Last OI": last_oi, "Len": hist_len, "Recent": " | ".join(recent), "Bootstrapped": bootstrapped, "PersistedFile": "yes" if persisted_exists else "no"})
    dbg_df = pd.DataFrame(dbg_rows)
    st.table(dbg_df)

    def breach_count(df):
        count = 0
        total = 0
        for m in OI_CHANGE_INTERVALS_MIN:
            col = f"OI %Chg ({m}m)"
            thr = PCT_CHANGE_THRESHOLDS[m]
            vals = df[col].dropna()
            total += len(vals)
            count += (vals.abs() > thr).sum()
        return count, total

    bc, bt = breach_count(calls_df)
    pc, pt = breach_count(puts_df)
    # show a compact alert summary
    if bc + pc > 0:
        st.warning(f"Threshold breaches — Calls: {bc}/{bt}, Puts: {pc}/{pt}")
    else:
        st.success(f"No %OI threshold breaches — Calls: {bc}/{bt}, Puts: {pc}/{pt}")

    # record UI processed time to power countdown next run
    st.session_state["ui_last_processed_ts"] = time.time()

else:
    st.info("Select a **Target Expiry** and click **Run / Refresh**.")

st.succsess("bye")
