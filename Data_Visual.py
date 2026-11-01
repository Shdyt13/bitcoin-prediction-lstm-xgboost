# ---------------------------------------------------------
# Copyright (c) Shdyt13
# ---------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# ==========================================
# === PRA-PEMROSESAN DATA ===
# ==========================================
def load_btc_data(filepath):
    """
    Membaca dan membersihkan dataset CSV multi-header hasil unduhan yfinance (BTC_2020_2026.csv).
    Mengembalikan DataFrame yang sudah diurutkan berdasarkan Tanggal dengan tipe numerik pada OHLCV.
    """
    if not os.path.exists(filepath):
        # Fallback jika dijalankan dari dalam sub-direktori
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "data", "BTC_2020_2026.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {filepath}")

    # Membaca baris pertama untuk mengambil nama kolom (Price, Adj Close, Close, High, Low, Open, Volume)
    header_cols = pd.read_csv(filepath, nrows=1).columns.tolist()
    header_cols[0] = "Date"

    # Melewati 3 baris metadata header bawaan yfinance
    df = pd.read_csv(filepath, skiprows=3, names=header_cols)

    # Parsing kolom tanggal dan pengurutan kronologis
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Konversi kolom OHLCV ke tipe numerik float
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return df


# ==========================================
# === VISUALISASI GRAFIK CANDLESTICK ===
# ==========================================
def plot_candlestick(df, resample_rule="W", title_suffix="(Weekly Aggregation)"):
    """
    Menampilkan grafik Candlestick beserta Moving Average (MA) dan Volume transaksi.
    Secara default menggunakan agregasi mingguan ('W') agar ~2.300+ hari data terlihat tajam dan jelas,
    namun tetap dapat di-zoom secara interaktif pada jendela Matplotlib.
    Set resample_rule=None jika ingin mem-plot seluruh candle harian secara langsung.
    """
    plot_df = df.copy()

    if resample_rule:
        plot_df = (
            plot_df.set_index("Date")
            .resample(resample_rule)
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna()
            .reset_index()
        )

    # Menghitung indikator Moving Average untuk memperjelas tren
    plot_df["MA20"] = plot_df["Close"].rolling(window=20).mean()
    plot_df["MA50"] = plot_df["Close"].rolling(window=50).mean()

    # Konversi tanggal ke format angka Matplotlib
    dates_num = mdates.date2num(plot_df["Date"])
    if len(dates_num) > 1:
        candle_width = (dates_num[1] - dates_num[0]) * 0.72
    else:
        candle_width = 0.6

    # Konfigurasi tema gelap (Doff Dark) yang senada dengan Dashboard UI
    bg_dark = "#121212"
    card_dark = "#1E1E1E"
    text_light = "#F5F5F5"
    grid_color = "#2E2E2E"
    color_bull = "#10B981"  # Hijau (Naik / Bullish)
    color_bear = "#EF4444"  # Merah (Turun / Bearish)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06},
        facecolor=bg_dark,
    )
    ax_price.set_facecolor(card_dark)
    ax_vol.set_facecolor(card_dark)

    # Mem-plot sumbu (wick/shadow) dan badan (body) Candlestick
    for d, open_p, high_p, low_p, close_p in zip(
        dates_num, plot_df["Open"], plot_df["High"], plot_df["Low"], plot_df["Close"]
    ):
        is_bullish = close_p >= open_p
        color = color_bull if is_bullish else color_bear

        # Garis High-Low (Sumbu / Wick)
        ax_price.plot([d, d], [low_p, high_p], color=color, linewidth=1.0, zorder=2)

        # Kotak Open-Close (Badan / Body Candle)
        lower = min(open_p, close_p)
        height = max(abs(close_p - open_p), (high_p - low_p) * 0.015)
        rect = Rectangle(
            (d - candle_width / 2.0, lower),
            candle_width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
        ax_price.add_patch(rect)

    # Mem-plot garis Moving Average
    ax_price.plot(
        dates_num, plot_df["MA20"],
        color="#38BDF8", linewidth=1.4, label="MA 20", alpha=0.9, zorder=4
    )
    ax_price.plot(
        dates_num, plot_df["MA50"],
        color="#F59E0B", linewidth=1.4, label="MA 50", alpha=0.9, zorder=4
    )

    # Mem-plot grafik batang Volume di panel bawah
    vol_colors = [
        color_bull if c >= o else color_bear
        for o, c in zip(plot_df["Open"], plot_df["Close"])
    ]
    ax_vol.bar(
        dates_num, plot_df["Volume"],
        width=candle_width, color=vol_colors, alpha=0.75, zorder=2
    )

    # Penataan Judul, Label, dan Grid
    ax_price.set_title(
        f"Bitcoin (BTC-USD) Candlestick Chart (2020 - 2026) {title_suffix}",
        color=text_light, fontsize=15, fontweight="bold", pad=14
    )
    ax_price.set_ylabel("Price (USD)", color=text_light, fontsize=11, fontweight="bold")
    ax_vol.set_ylabel("Volume", color=text_light, fontsize=11, fontweight="bold")
    ax_vol.set_xlabel("Date", color=text_light, fontsize=11, fontweight="bold")

    for ax in (ax_price, ax_vol):
        ax.grid(True, linestyle="--", linewidth=0.6, color=grid_color, alpha=0.8, zorder=1)
        ax.tick_params(colors=text_light, labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # Format sumbu X sebagai tanggal (Tahun-Bulan)
    ax_vol.xaxis_date()
    ax_vol.xaxis.set_major_locator(mdates.YearLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_vol.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))

    # Legenda pada panel harga
    legend = ax_price.legend(loc="upper left", facecolor=bg_dark, edgecolor="#333333", fontsize=10)
    for text in legend.get_texts():
        text.set_color(text_light)

    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.06)
    output_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BTC_2020_2026_Candlestick.png")
    plt.savefig(output_img, dpi=200, facecolor=bg_dark, bbox_inches="tight")
    print(f"Grafik Candlestick berhasil disimpan ke: {output_img}")
    plt.show()


if __name__ == "__main__":
    csv_path = os.path.join("backend", "data", "BTC_2020_2026.csv")
    print(f"Memuat data dari '{csv_path}'...")
    btc_df = load_btc_data(csv_path)
    print(
        f"Data berhasil dimuat: {len(btc_df)} baris "
        f"({btc_df['Date'].min().strftime('%Y-%m-%d')} s/d {btc_df['Date'].max().strftime('%Y-%m-%d')})"
    )

    # Menampilkan Candlestick (Gunakan resample_rule='W' untuk mingguan, atau None untuk harian)
    plot_candlestick(btc_df, resample_rule="W", title_suffix="(Weekly Candle)")
