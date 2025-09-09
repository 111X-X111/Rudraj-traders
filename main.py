#!/usr/bin/env python3
# main.py — Streamlit OI tracker (live via kite.quote) — full corrected file
#
# Install:
#   pip install streamlit kiteconnect pandas pytz
#
# Run:
#   streamlit run main.py

import os
import time
from datetime import datetime, date, timedelta, timezone, time as dtime

import pandas as pd
import pytz
import streamlit as st
from kiteconnect import KiteConnect

# Prefer st_autorefresh when available
try:
    from streamlit import st_autorefresh  # type: ignore
    HAVE_ST_AUTOREFRESH = True
except Exception:
    HAVE_ST_AUTOREFRESH = False

# ---------- Config ----------
HISTORICAL_MAX_MINUTES = 120
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30)
PCT_CHANGE_THRESHOLDS = {5: 8.0, 10: 10.0, 15: 15.0, 30: 25.0}
IST = pytz.timezone("Asia/Kolkata")

EXCHANGE_NFO = KiteConnect.EXCHANGE_NFO
EXCHANGE_LTP = KiteConnect.EXCHANGE_NSE

API_KEY = "afzia78cwraaod5x"
API_SECRET = "b527807j5ilcndjp5u2jhu9znrjxz35e"

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
    except Exception as e:
        st.error(f"Error fetching LTP for {underlying_sym}: {e}")
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

# ---------- IMPORTANT: function that was missing ----------
def get_relevant_option_details(instruments: list, atm: float, expiry_dt: date,
                                strike_diff: int, levels: int,
                                underlying_prefix: str, symbol_prefix: str):
    """
    Find option contracts around ATM for the selected expiry.
    Returns dict keyed like "atm_ce", "otm1_pe", etc. with tradingsymbol and instrument_token.
    """
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

# ---------- History and live quote handling ----------
def _ensure_session_struct():
    if "oi_history" not in st.session_state:
        st.session_state["oi_history"] = {}  # key -> list of (utc_dt, oi)
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = 0.0
    if "last_quote_success" not in st.session_state:
        st.session_state["last_quote_success"] = None
    if "refresh_count" not in st.session_state:
        st.session_state["refresh_count"] = 0

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
    except Exception as e:
        st.warning(f"quote() failed: {e}")
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
        st.session_state["oi_history"][key] = [t for t in hist if t[0] >= cutoff]

        if oi is not None:
            any_success = True

    if any_success:
        st.session_state["last_quote_success"] = now_utc
    st.session_state["refresh_count"] += 1

def fetch_last_oi_at_close(kite: KiteConnect, details: dict):
    _ensure_session_struct()
    today_ist = datetime.now(IST).date()
    close_utc = market_close_datetime_utc(today_ist)
    from_utc = close_utc - timedelta(minutes=max(HISTORICAL_MAX_MINUTES, max(OI_CHANGE_INTERVALS_MIN)))

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
        except Exception as e:
            st.warning(f"historical_data failed for token {tok}: {e}")

def compute_pct_changes_from_history(details: dict, intervals=(5,10,15,30)):
    _ensure_session_struct()
    out = {}
    now_utc = datetime.now(timezone.utc)
    hist_all = st.session_state.get("oi_history", {})

    for key, d in details.items():
        hist = hist_all.get(key, [])
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
            for t, o in reversed(hist):
                if t <= past_target and o is not None:
                    past_oi = o
                    break
            pct = None
            if past_oi not in (None, 0) and latest_oi not in (None,):
                try:
                    pct = (latest_oi - past_oi) / past_oi * 100.0
                except Exception:
                    pct = None
            out[key][f"pct_diff_{m}m"] = pct
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

# ---------- Streamlit UI ----------
st.set_page_config(page_title="OI Tracker (Live OI via quote)", layout="wide")

if not check_password():
    st.stop()

st.title("📊 OI Tracker — Live OI (kite.quote) — corrected")

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
    refresh = st.number_input("Refresh seconds", min_value=5, max_value=300, value=30, step=5)
    autorun = st.checkbox("Auto-refresh", value=False)
    go = st.button("Run / Refresh")

market_closed = is_market_closed_now()

# Auto-refresh
if autorun and not market_closed:
    if HAVE_ST_AUTOREFRESH:
        st_autorefresh(interval=refresh * 1000, key="autorefresh_counter")
    else:
        last = st.session_state.get("last_refresh", 0.0)
        now = time.time()
        if now - last > refresh:
            st.session_state["last_refresh"] = now
            try:
                st.experimental_rerun()
            except Exception:
                pass

if "access_token" not in st.session_state:
    st.info("Generate Access Token in the sidebar to begin.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(st.session_state["access_token"])

try:
    prof = kite.profile()
    st.success(f"Connected: {prof.get('user_id')} — {prof.get('user_name')}")
except Exception as e:
    st.error(f"Login failed: {e}")
    st.stop()

@st.cache_data(ttl=600)
def load_instruments(nfo_exchange: str):
    return kite.instruments(nfo_exchange)

instruments = load_instruments(EXCHANGE_NFO)
under_prefix = underlying.split(" ")[0].upper()
expiries = list_expiries(instruments, under_prefix)
if not expiries:
    st.error("No future expiries found.")
    st.stop()

exp_strs = [ex.strftime("%d-%b-%Y (%a)") for ex in expiries]
sel_idx = st.sidebar.selectbox("Target Expiry", range(len(expiries)), format_func=lambda i: exp_strs[i], index=0)
target_expiry = expiries[sel_idx]
symbol_prefix = symbol_prefix_for_expiry(instruments, under_prefix, target_expiry)

if go or autorun or market_closed:
    atm, ltp = get_atm_strike(kite, underlying, EXCHANGE_LTP, strike_diff)
    if atm is None:
        st.error("Unable to compute ATM.")
        st.stop()

    st.subheader(f"{underlying} — Spot: {ltp:.2f} | ATM: {int(atm)} | Expiry: {target_expiry.strftime('%d-%b-%Y')}")
    _ensure_session_struct()
    last_q = st.session_state.get("last_quote_success")
    if last_q:
        st.caption(f"Last successful quote (IST): {last_q.astimezone(IST).strftime('%d-%b-%Y %H:%M:%S')}")
    else:
        st.caption("No successful quote yet this session.")

    if market_closed:
        st.info("Market closed — showing last available values (no live polling).")
    else:
        if autorun:
            st.caption(f"Auto-refresh ON — interval {refresh}s.")
        else:
            st.caption("Auto-refresh OFF — click Run / Refresh to fetch once.")

    details = get_relevant_option_details(instruments, atm, target_expiry, strike_diff, levels, under_prefix, symbol_prefix)
    if not details:
        st.warning("No matching option contracts found around ATM for selected expiry.")
        st.stop()

    if not market_closed:
        update_live_oi_from_quote(kite, details)
    else:
        need_fetch = any(not st.session_state["oi_history"].get(k) for k in details.keys())
        if need_fetch:
            fetch_last_oi_at_close(kite, details)

    report = compute_pct_changes_from_history(details, intervals=OI_CHANGE_INTERVALS_MIN)

    calls_df = build_table(report, details, atm, strike_diff, levels, is_call=True)
    puts_df  = build_table(report, details, atm, strike_diff, levels, is_call=False)

    fmt_cols = {"Latest OI": lambda v: safe_fmt(v, "{:,.0f}") if v not in (None, "") else ""}
    for m in OI_CHANGE_INTERVALS_MIN:
        col = f"OI %Chg ({m}m)"
        fmt_cols[col] = (lambda f: (lambda v: safe_fmt(v, f)))("{:+.2f}%")

    st.markdown("### Calls (ΔOI %)")
    st.dataframe(calls_df.style.format(fmt_cols))

    st.markdown("### Puts (ΔOI %)")
    st.dataframe(puts_df.style.format(fmt_cols))

    # Debug panel
    st.markdown("### Debug: history preview (symbol → last_ts IST, last_oi, history_len, recent entries)")
    dbg_rows = []
    for key, d in details.items():
        hist = st.session_state["oi_history"].get(key, [])
        last_ts = hist[-1][0].astimezone(IST).strftime("%H:%M:%S") if hist else ""
        last_oi = hist[-1][1] if hist else None
        hist_len = len(hist)
        recent = []
        for t, o in hist[-3:]:
            recent.append(f"{t.astimezone(IST).strftime('%H:%M:%S')}:{(o if o is not None else 'None')}")
        dbg_rows.append({"Symbol": d.get("tradingsymbol", ""), "Last TS(IST)": last_ts, "Last OI": last_oi, "Len": hist_len, "Recent": " | ".join(recent)})
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
    st.info(f"Threshold breaches — Calls: {bc}/{bt}, Puts: {pc}/{pt}")

else:
    st.info("Select a **Target Expiry** and click **Run / Refresh**.")
