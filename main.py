#!/usr/bin/env python3
# main.py — Streamlit OI tracker with local persistence of minute OI history
#
# Install:
#   pip install streamlit kiteconnect pandas pytz
#
# Run:
#   streamlit run main.py

import os
import json
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
HISTORY_FILE = "oi_history.json"
HISTORICAL_MAX_MINUTES = 120
BOOTSTRAP_MINUTES = 60        # minutes fetched on startup to bootstrap session history
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30)
PCT_CHANGE_THRESHOLDS = {5: 8.0, 10: 10.0, 15: 15.0, 30: 25.0}
IST = pytz.timezone("Asia/Kolkata")

EXCHANGE_NFO = KiteConnect.EXCHANGE_NFO
EXCHANGE_LTP = KiteConnect.EXCHANGE_NSE

API_KEY = "afzia78cwraaod5x"
API_SECRET = "b527807j5ilcndjp5u2jhu9znrjxz35e"

# ---------- Utilities: persistence ----------
def _dt_to_iso(dt):
    if dt is None:
        return None
    # ensure aware in UTC
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
    """
    Load persisted history from JSON. Format:
      { key: [ [iso_ts, oi_or_null], ... ], ... }
    Returns dict: key -> list of (utc_dt, oi)
    """
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
        # keep only last HISTORICAL_MAX_MINUTES window
        if lst:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=HISTORICAL_MAX_MINUTES)
            lst = [t for t in lst if t[0] is not None and t[0] >= cutoff]
        out[key] = lst
    return out

def save_persisted_history(history: dict, path=HISTORY_FILE):
    """
    Save history dict (key -> list of (utc_dt, oi)) to JSON atomically.
    """
    tmp = path + ".tmp"
    raw = {}
    for key, lst in history.items():
        # if timestamps are None, skip them
        arr = []
        for t, oi in lst:
            arr.append([_dt_to_iso(t), oi])
        raw[key] = arr
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # best-effort, ignore persistence errors
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
    except Exception as e:
        # error fetching LTP
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
        # start by loading persisted history (best-effort)
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

def persist_history_after_change():
    """Save the st.session_state['oi_history'] to disk."""
    try:
        # prune each key to HISTORICAL_MAX_MINUTES before saving
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
    """
    Poll kite.quote for given option tradingsymbols.
    Write canonical timestamp = now_utc for all quote entries for consistency.
    Persist after update.
    """
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
    except Exception as e:
        # network / auth / rate limit failure => offline mode
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
        # prune
        cutoff = now_utc - timedelta(minutes=max(HISTORICAL_MAX_MINUTES, max(OI_CHANGE_INTERVALS_MIN)))
        st.session_state["oi_history"][key] = [t for t in hist if t[0] is not None and t[0] >= cutoff]

        if oi is not None:
            any_success = True

    if any_success:
        st.session_state["last_quote_success"] = now_utc
    st.session_state["refresh_count"] += 1
    # persist to disk
    persist_history_after_change()

def fetch_last_oi_at_close(kite: KiteConnect, details: dict):
    _ensure_session_struct()
    today_ist = datetime.now(IST).date()
    close_utc = market_close_datetime_utc(today_ist)
    from_utc = close_utc - timedelta(minutes=max(HISTORICAL_MAX_MINUTES, max(OI_CHANGE_INTERVALS_MIN)))

    try:
        # test connectivity
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
    """
    Fetch minute candles with oi=True for the last `minutes` and store them to session history.
    Uses persistence so subsequent restarts keep this data.
    """
    _ensure_session_struct()
    now_utc = datetime.now(timezone.utc)
    to_dt = now_utc
    from_dt = to_dt - timedelta(minutes=minutes)

    # check connectivity
    try:
        _ = kite.profile()
        st.session_state["offline"] = False
    except Exception:
        st.session_state["offline"] = True
        return

    for key, d in details.items():
        tok = d.get("instrument_token")
        if not tok:
            # mark as attempted to avoid repeated failing calls
            st.session_state["bootstrap_done_for"].add(key)
            continue
        # skip if already bootstraped this session
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
            # merge into session history
            existing = st.session_state["oi_history"].get(key, [])
            # combine and deduplicate by timestamp
            combined = existing + hist_list
            # keep unique by timestamp, sorted
            combined_map = {}
            for t,o in combined:
                if t is None:
                    continue
                combined_map[t.isoformat()] = (t, o)
            combined_sorted = [combined_map[k] for k in sorted(combined_map.keys())]
            # prune to HISTORICAL_MAX_MINUTES window
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=HISTORICAL_MAX_MINUTES)
            combined_sorted = [t for t in combined_sorted if t[0] >= cutoff]
            st.session_state["oi_history"][key] = combined_sorted
            st.session_state["bootstrap_done_for"].add(key)
            if combined_sorted:
                st.session_state["last_quote_success"] = combined_sorted[-1][0]
        except Exception:
            st.session_state["bootstrap_done_for"].add(key)
            continue
    # persist merged history
    persist_history_after_change()

def compute_pct_changes_from_history(details: dict, intervals=(5,10,15,30)):
    """
    Return dict keyed by symbol key. For each interval m:
      - pct_diff_{m}m : percentage or None
      - pct_used_earliest_{m}m : True if earliest fallback was used
    """
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

# ---------- Streamlit UI ----------
st.set_page_config(page_title="OI Tracker (Live OI via quote + persistence)", layout="wide")

if not check_password():
    st.stop()

st.title("📊 OI Tracker — Live OI + local persistence (shows lookbacks after restart/offline)")

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

# ensure session history loaded
_ensure_session_struct()

if "access_token" not in st.session_state:
    st.info("Generate Access Token in the sidebar to begin.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(st.session_state["access_token"])

# Check login; if fails we go to offline mode but still show persisted data
try:
    prof = kite.profile()
    st.success(f"Connected: {prof.get('user_id')} — {prof.get('user_name')}")
    st.session_state["offline"] = False
except Exception:
    st.warning("Could not verify Kite login — working in OFFLINE mode using persisted history if available.")
    st.session_state["offline"] = True

# Load instruments (cached)
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

    # Bootstrapping: if market open & not offline, ensure we have history to cover lookbacks
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

    # Update live quotes (if possible)
    if not market_closed and not st.session_state.get("offline"):
        update_live_oi_from_quote(kite, details)
    else:
        # After close or offline: ensure we have a last-close value if no persisted/bootstrapped history
        need_fetch = False
        for key in details.keys():
            if not st.session_state["oi_history"].get(key):
                need_fetch = True
                break
        if need_fetch and not st.session_state.get("offline"):
            fetch_last_oi_at_close(kite, details)

    # If offline and we have persisted history, we'll compute using that.
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
    st.info(f"Threshold breaches — Calls: {bc}/{bt}, Puts: {pc}/{pt}")

else:
    st.info("Select a **Target Expiry** and click **Run / Refresh**.")
