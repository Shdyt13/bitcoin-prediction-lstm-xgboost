import pandas as pd


class DataFormatError(ValueError):
    """Raised when the input CSV is missing expected columns or has no usable rows."""


def _find_column(df, candidates):
    """Return the first column name from `candidates` that exists in df, or raise."""
    for name in candidates:
        if name in df.columns:
            return name
    raise DataFormatError(
        f"Could not find any of the expected columns {candidates}. "
        f"Columns found in file: {list(df.columns)}"
    )


def _clean_currency_column(series):
    """
    Strip currency symbols/letters/spaces and thousand separators, then
    convert to float. Unparsable values become NaN instead of raising,
    so a single malformed row doesn't crash the whole pipeline.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)  # drop "Rp", spaces, stray letters
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_bitcoin_data(filepath):
    """
    Read a Bitcoin price CSV and clean up its currency formatting.

    Returns (df, df_bersih, kolom_tanggal, kolom_close):
    - df: the full parsed dataframe, sorted by date
    - df_bersih: cleaned Open/Close columns only, forward-filled, NaN-free
    - kolom_tanggal, kolom_close: resolved column names, for plotting

    Raises:
        FileNotFoundError: if `filepath` doesn't exist.
        DataFormatError: if the expected date/price columns are missing,
            the file is empty, or no usable rows remain after cleaning.
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise DataFormatError(f"Could not parse CSV at '{filepath}': {exc}") from exc

    if df.empty:
        raise DataFormatError(f"CSV at '{filepath}' has no rows.")

    kolom_tanggal = _find_column(df, ["Tanggal", "Date"])
    kolom_open = _find_column(df, ["Buka*", "Open"])
    kolom_close = _find_column(df, ["Tutup**", "Close"])

    df[kolom_tanggal] = pd.to_datetime(df[kolom_tanggal], format="mixed", errors="coerce")
    n_bad_dates = int(df[kolom_tanggal].isna().sum())
    if n_bad_dates:
        df = df.dropna(subset=[kolom_tanggal])
        print(f"Peringatan: {n_bad_dates} baris dengan tanggal tak terbaca dibuang dari '{filepath}'.")

    df = df.sort_values(kolom_tanggal).reset_index(drop=True)

    for kolom in (kolom_open, kolom_close):
        df[kolom] = _clean_currency_column(df[kolom])

    df_bersih = df[[kolom_open, kolom_close]].ffill().dropna()

    if df_bersih.empty:
        raise DataFormatError(
            f"Tidak ada baris valid tersisa setelah pembersihan di '{filepath}'. "
            f"Periksa format kolom '{kolom_open}'/'{kolom_close}'."
        )

    return df, df_bersih, kolom_tanggal, kolom_close