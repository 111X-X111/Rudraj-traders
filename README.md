  # 📊 OI Tracker — Zerodha KiteConnect + Streamlit

This project is a **Streamlit dashboard** for tracking **Open Interest (OI) changes** around ATM strikes, with expiry selection.  
It integrates with **Zerodha KiteConnect API** to fetch live option chain and OI data.

---

## 🚀 Features
- Secure login with password prompt  
- Zerodha Kite access token generator (via request token)  
- ATM strike auto-detection  
- Expiry selection (drop-down)  
- Real-time OI % change for Calls and Puts across intervals (5m, 10m, 15m, 30m)  
- Threshold breach alerts  

---

## 🛠 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/oi-tracker.git
cd oi-tracker
pip install -r requirements.txt
