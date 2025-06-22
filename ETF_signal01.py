import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import matplotlib.dates as mdates
import sys
print(sys.executable)

# --- 參數設定（可調整） ---
WEIGHT_PREMIUM = 0.5  # 折溢價率占比
WEIGHT_CNYES = 0.1    # 鉅亨新聞權重
WEIGHT_MEGA = 0.1     # 兆豐新聞權重
WEIGHT_PTT = 0.1      # PTT 輿情權重
WEIGHT_VIX = 0.2      # VIX 占比權重

# --- 設定路徑 ---
base_path = "C:/Users/winni/PycharmProjects/PythonProject1/ETF_signal/ETF_data"

# --- 設定字體避免中文亂碼 ---
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 讀取資料 ---
df_PremiumDiscount = pd.read_csv("MoneyDJ_ETF_PremiumDiscount_00757.csv")
df_PremiumDiscount.set_index("交易日期", inplace=True)
df_PremiumDiscount.index = pd.to_datetime(df_PremiumDiscount.index)

df_sentiment = pd.read_csv(os.path.join(base_path, "sentiment_result.csv"))
df_sentiment["日期"] = pd.to_datetime(df_sentiment["日期"]).dt.date
df_sentiment.set_index("日期", inplace=True)

df_VIX = pd.read_csv("vix_daily.csv")
df_VIX["Date"] = pd.to_datetime(df_VIX["Date"])
df_VIX.set_index("Date", inplace=True)

# --- 建立每日資料表 ---
start_date = "2020-01-01"
end_date = "2025-05-31"
all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

result = pd.DataFrame({
    "Date": all_dates,
    "is_trading_day": pd.Series([pd.NA] * len(all_dates), dtype="boolean"),
    "市價": pd.Series([pd.NA] * len(all_dates), dtype="object"),
    "折溢價利率(%)": pd.Series([pd.NA] * len(all_dates), dtype="object"),
    "折溢價利率分數": pd.Series([np.nan] * len(all_dates), dtype="float"),
    "新聞輿情分數": pd.Series([np.nan] * len(all_dates), dtype="float"),
    "VIX": pd.Series([pd.NA] * len(all_dates), dtype="object"),
    "指數綜合分數": pd.Series([pd.NA] * len(all_dates), dtype="Int64"),
})

# --- 分數計算函式 ---
def classify_score_index(vix):
    if pd.isna(vix): return pd.NA
    if vix > 22.72: return 1
    elif vix < 17.12: return -1
    else: return 0

def score_PremiumDiscount_weighted(p):
    if pd.isna(p): return pd.NA
    try:
        val = float(p.replace('%', '')) if isinstance(p, str) else float(p)
    except:
        return pd.NA

    if val <= -0.49:
        return 0.5
    elif val <= -0.16:
        return 0.25
    elif val <= 0.09:
        return 0
    elif val <= 0.38:
        return -0.25
    else:
        return -0.5

# --- 每日填值 ---
for d in all_dates:
    d_date = d.date()
    result.loc[result["Date"] == d, "市價"] = df_PremiumDiscount["市價"].get(d, pd.NA)
    rate = df_PremiumDiscount["折溢價利率(%)"].get(d, pd.NA)
    result.loc[result["Date"] == d, "折溢價利率(%)"] = rate
    result.loc[result["Date"] == d, "折溢價利率分數"] = score_PremiumDiscount_weighted(rate)

    cnyes = df_sentiment["鉅亨_左側情緒分類"].get(d_date, pd.NA)
    mega = df_sentiment["兆豐_左側情緒分類"].get(d_date, pd.NA)
    ptt = df_sentiment["PTT_左側情緒分類"].get(d_date, pd.NA)
    score = (
        (float(cnyes) if not pd.isna(cnyes) else 0) * WEIGHT_CNYES +
        (float(mega) if not pd.isna(mega) else 0) * WEIGHT_MEGA +
        (float(ptt) if not pd.isna(ptt) else 0) * WEIGHT_PTT
    )
    result.loc[result["Date"] == d, "新聞輿情分數"] = score

    vix = df_VIX["Close"].get(d, pd.NA)
    result.loc[result["Date"] == d, "VIX"] = vix
    result.loc[result["Date"] == d, "指數綜合分數"] = classify_score_index(vix)

# --- 計算總分與燈號 ---
result["總分"] = (
    result["折溢價利率分數"].astype("float") +
    result["新聞輿情分數"].astype("float") +
    result["指數綜合分數"].astype("float") * WEIGHT_VIX
)

def classify_signal(score):
    if pd.isna(score): return pd.NA
    if score >= 0.8: return "深綠燈"
    elif 0.2 <= score < 0.8: return "淺綠燈"
    elif -0.4 < score < 0.2: return "黃燈"
    elif -0.8 < score <= -0.4: return "淺紅燈"
    elif score <= -0.8: return "紅燈"
    else: return pd.NA

result["燈號"] = result["總分"].apply(classify_signal)

# --- 匯出簡易結果 ---
result.to_csv("簡易回測結果.csv", index=False, encoding="utf-8-sig")
print("✅ 已輸出簡易回測結果.csv")

# --- 動態互動圖 ---
plot_df = result[result["燈號"] != "黃燈"].dropna(subset=["市價"])
plot_df["市價"] = pd.to_numeric(plot_df["市價"], errors="coerce")

fig = px.scatter(plot_df, x="Date", y="市價", color="燈號",
                 title="互動式：市價與燈號標記（不含黃燈）",
                 hover_data=["總分", "折溢價利率(%)"],
                 color_discrete_map={
                     "紅燈": "red",
                     "淺紅燈": "salmon",
                     "淺綠燈": "lightgreen",
                     "深綠燈": "green"
                 })
fig.add_scatter(x=result["Date"], y=result["市價"], mode="lines", name="收盤價",
                line=dict(color="blue"))
fig.write_html("signal_plot_interactive.html")
print("✅ 互動圖已儲存為 signal_plot_interactive.html")
