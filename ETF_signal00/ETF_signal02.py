import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import matplotlib.dates as mdates
import plotly.graph_objects as go

# --- 參數設定 ---
STOCK_ID = "00646"
WEIGHT_PREMIUM = 0.5  # 折溢價率占比
WEIGHT_CNYES = 0.1    # 鉅亨新聞權重
WEIGHT_MEGA = 0.1     # 兆豐新聞權重
WEIGHT_PTT = 0.1      # PTT 輿情權重
WEIGHT_VIX = 0.2      # VIX 占比權重

# --- 設定路徑 ---
base_path = os.path.dirname(__file__)
filename = f"MoneyDJ_ETF_PremiumDiscount_{STOCK_ID}.csv"

# --- 設定字體避免中文亂碼 ---
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 動態 z-score 分數函式 ---
def score_PremiumDiscount_z_dynamic(df, window=30):
    df = df.copy()
    df["折溢價率"] = df["折溢價利率(%)"].str.replace('%', '').astype(float)
    df["z_mean"] = df["折溢價率"].rolling(window=window).mean()
    df["z_std"] = df["折溢價率"].rolling(window=window).std()
    df["z_score"] = (df["折溢價率"] - df["z_mean"]) / df["z_std"]

    scores = []
    no_positive_days = 0
    for z in df["z_score"]:
        score = 0
        if pd.isna(z):
            scores.append(pd.NA)
            continue

        if z <= -1.2:
            score = 0.5
        elif z <= -0.3:
            score = 0.25
        elif z >= 1.2:
            score = -0.5
        elif z >= 0.3:
            score = -0.25
        else:
            score = 0

        # 擴大補強條件：若連續 30 天未出現 score >= 0.25 則強制補 +0.25
        if score >= 0.25:
            no_positive_days = 0
        else:
            no_positive_days += 1
            if no_positive_days >= 30 and -0.3 <= z <= 0.3:
                score = 0.25

        scores.append(score)

    df["折溢價分數"] = scores
    return df

# --- 讀取資料 ---
df_PremiumDiscount = pd.read_csv(os.path.join(base_path, filename))
df_PremiumDiscount["交易日期"] = pd.to_datetime(df_PremiumDiscount["交易日期"])
df_PremiumDiscount.set_index("交易日期", inplace=True)
df_PremiumDiscount = score_PremiumDiscount_z_dynamic(df_PremiumDiscount)

# --- 讀取其他資料 ---
df_sentiment = pd.read_csv(os.path.join(base_path, "sentiment_result.csv"))
df_sentiment["日期"] = pd.to_datetime(df_sentiment["日期"]).dt.date
df_sentiment.set_index("日期", inplace=True)

df_VIX = pd.read_csv(os.path.join(base_path, "vix_daily.csv"))
df_VIX["Date"] = pd.to_datetime(df_VIX["Date"])
df_VIX.set_index("Date", inplace=True)

# --- 建立每日資料表 ---
start_date = "2020-01-01"
end_date = "2025-05-31"
all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

result = pd.DataFrame({
    "Date": all_dates,
    "市價": pd.Series([pd.NA] * len(all_dates), dtype="object"),
    "折溢價利率(%)": pd.Series([pd.NA] * len(all_dates), dtype="object"),
    "折溢價分數": pd.Series([np.nan] * len(all_dates), dtype="float"),
    "新聞輿情分數": pd.Series([np.nan] * len(all_dates), dtype="float"),
    "VIX": pd.Series([pd.NA] * len(all_dates), dtype="object"),
    "指數綜合分數": pd.Series([np.nan] * len(all_dates), dtype="float"),
})

# --- 分數計算函式 ---
def classify_score_index(vix):
    if pd.isna(vix):
        return pd.NA
    if vix >= 30: return 1
    elif 25 <= vix < 30:
        return 0.5
    elif 20 <= vix < 25:
        return 0
    elif 15 <= vix < 20:
        return -0.5
    elif vix < 15: return -1
    else:
        return pd.NA  # 防守性處理

# --- 每日填值 ---
for d in all_dates:
    d_date = d.date()
    result.loc[result["Date"] == d, "市價"] = df_PremiumDiscount["市價"].get(d, pd.NA)
    result.loc[result["Date"] == d, "折溢價利率(%)"] = df_PremiumDiscount["折溢價利率(%)"].get(d, pd.NA)
    result.loc[result["Date"] == d, "折溢價分數"] = df_PremiumDiscount["折溢價分數"].get(d, pd.NA)

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
    result["折溢價分數"].astype("float") +
    result["新聞輿情分數"].astype("float") +
    result["指數綜合分數"].astype("float") * WEIGHT_VIX
)

def classify_signal(score):
    if pd.isna(score): return pd.NA
    if score >= 0.5: return "深綠燈"
    elif 0.2 <= score < 0.5: return "淺綠燈"
    elif -0.5 < score < 0.2: return "黃燈"
    elif -0.7 < score <= -0.5: return "淺紅燈"
    elif score <= -0.7: return "紅燈"
    else: return pd.NA

result["燈號"] = result["總分"].apply(classify_signal)

# --- 抑制連續紅燈/綠燈疲乏機制 ---
燈號_series = result["燈號"].tolist()
總分_series = result["總分"].tolist()

for i in range(3, len(result)):
    # --- 安全檢查：避免比對 NA 值 ---
    if all(pd.notna(燈號_series[i - j]) for j in range(4)):
        # 淺紅燈疲乏條件
        if (
            燈號_series[i - 3] == "淺紅燈" and
            燈號_series[i - 2] == "淺紅燈" and
            燈號_series[i - 1] == "淺紅燈" and
            燈號_series[i] == "淺紅燈"
        ):
            min_score = min(總分_series[i - 3:i])
            if pd.notna(總分_series[i]) and 總分_series[i] >= min_score:
                燈號_series[i] = "黃燈"

        # 淺綠燈疲乏條件
        if (
            燈號_series[i - 3] == "淺綠燈" and
            燈號_series[i - 2] == "淺綠燈" and
            燈號_series[i - 1] == "淺綠燈" and
            燈號_series[i] == "淺綠燈"
        ):
            max_score = max(總分_series[i - 3:i])
            if pd.notna(總分_series[i]) and 總分_series[i] <= max_score:
                燈號_series[i] = "黃燈"

# 更新結果回 DataFrame
result["燈號"] = 燈號_series

# --- 匯出結果 ---
result.to_csv(f"ETF_signal_{STOCK_ID}.csv", index=False, encoding="utf-8-sig")
print("✅ ETF_signal.csv")

# --- 動態互動圖 ---
plot_df = result[(result["燈號"].notna()) & (result["燈號"] != "黃燈")].dropna(subset=["市價"])
plot_df["市價"] = pd.to_numeric(plot_df["市價"], errors="coerce")

# --- 燈號色碼設定 ---
燈號色碼 = {
    "紅燈": "#B00000",
    "淺紅燈": "salmon",
    "淺綠燈": "lightgreen",
    "深綠燈": "#008000"
}

# --- 建立互動圖物件 ---
fig = go.Figure()

# 1️⃣ 先畫收盤價線（lightsteelblue，降低干擾）
fig.add_trace(go.Scatter(
    x=result["Date"],
    y=pd.to_numeric(result["市價"], errors="coerce"),
    mode="lines",
    name="收盤價",
    line=dict(color="blue")
))

# 2️⃣ 再畫各類燈號的點（依序分組畫）
for light in plot_df["燈號"].unique():
    df_sub = plot_df[plot_df["燈號"] == light]
    fig.add_trace(go.Scatter(
        x=df_sub["Date"],
        y=df_sub["市價"],
        mode="markers",
        name=light,
        marker=dict(
            color=燈號色碼.get(light, "gray"),
            size=12 if light == "深綠燈" else 8,
            line=dict(color="white", width=1)
        ),
        hovertemplate=(
            f"<b>{light}</b><br>" +
            "日期: %{x}<br>" +
            "市價: %{y}<br>" +
            "總分: %{customdata[0]}<br>" +
            "折溢價率: %{customdata[1]}%"
        ),
        customdata=df_sub[["總分", "折溢價利率(%)"]].values
    ))

# --- 整體外觀調整 ---
fig.update_layout(
    title=f"互動式：市價與燈號標記（不含黃燈） - {STOCK_ID}",
    xaxis_title="日期",
    yaxis_title="市價",
    legend_title="燈號",
    hovermode="closest"
)

# --- 儲存 HTML ---
fig.write_html(f"signal_plot_interactive_{STOCK_ID}.html")
print("✅ 互動圖已儲存為 signal_plot_interactive.html")