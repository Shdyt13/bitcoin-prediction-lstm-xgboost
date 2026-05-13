import pandas as pd

def clean_bitcoin_data(filepath):
    """
    Membaca dataset CSV dan membersihkan format mata uang.
    Mengembalikan dataframe utuh (df), dataframe bersih (df_bersih),
    dan nama kolom untuk kemudahan plotting.
    """
    df = pd.read_csv(filepath, sep=None, engine='python')
    kolom_tanggal = 'Tanggal' if 'Tanggal' in df.columns else 'Date'
    kolom_open = 'Buka*' if 'Buka*' in df.columns else 'Open'
    kolom_close = 'Tutup**' if 'Tutup**' in df.columns else 'Close'
    
    df[kolom_tanggal] = pd.to_datetime(df[kolom_tanggal], format='mixed')
    
    df = df.sort_values(kolom_tanggal)
    
    # Membersihkan format mata uang
    df[kolom_open] = df[kolom_open].astype(str).str.replace('Rp ', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df[kolom_close] = df[kolom_close].astype(str).str.replace('Rp ', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # Ambil kolom fitur saja dan hapus NaN
    df_bersih = df[[kolom_open, kolom_close]].ffill().dropna()
    
    return df, df_bersih, kolom_tanggal, kolom_close