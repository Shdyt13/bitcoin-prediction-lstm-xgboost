import yfinance as yf

data = yf.download(
    "BTC-USD",
    start="2020-01-01",
    end="2026-05-21",
    auto_adjust=False
)

print(data.head())

data.to_csv("BTC_2020_2026.csv")