# 📈 NVDA Forecast — Streamlit App

A sleek, dark-mode **Streamlit** dashboard that predicts **NVIDIA’s next value** using a pre-trained **LightGBM** model. It also shows simple multi-ticker trends and a correlation heatmap for quick intuition.

> ✅ This repo is **Streamlit-only**. The model loads **locally** from `streamlit/models/` — no separate API required.

---

## 🚀 Quick Start (Local)

**Requirements:** Python **3.11** (see `runtime.txt`) and `pip`.

```bash
# 1) Clone
git clone https://github.com/<your-username>/stock_prediction.git
cd stock_prediction

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run streamlit/app.py
