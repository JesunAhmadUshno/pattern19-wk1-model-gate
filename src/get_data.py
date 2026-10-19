"""Fetch the public UCI Default of Credit Card Clients dataset and
normalize it to data/credit.csv. The dataset is public and old; see
records/data_card.md (generated) for provenance. Run once."""
import io
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_CSV  # noqa: E402

URL = ("https://archive.ics.uci.edu/static/public/350/"
       "default+of+credit+card+clients.zip")


def main():
    if os.path.exists(DATA_CSV):
        print(f"data already present: {DATA_CSV}")
        return
    import pandas as pd

    print(f"downloading {URL}")
    raw = urllib.request.urlopen(URL, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))]
    if not names:
        raise SystemExit(f"no spreadsheet in archive: {zf.namelist()}")
    blob = zf.read(names[0])
    # The classic sheet carries a two-row header: row 0 is X1..X23, Y.
    df = pd.read_excel(io.BytesIO(blob), header=1)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("default") or lc == "y":
            rename[c] = "default_next_month"
    df = df.rename(columns=rename)
    if "default_next_month" not in df.columns:
        raise SystemExit(f"target column not found in {list(df.columns)}")
    df = df.dropna(subset=["default_next_month"])
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"wrote {DATA_CSV}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
