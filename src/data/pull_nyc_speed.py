import argparse
from pathlib import Path
import requests
import pandas as pd

# NYC DOT Traffic Speeds dataset on NYC Open Data
# dataset page: https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds-NBE/i4gi-tjb9
BASE = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--out", default="data/raw/nyc_speeds.json")
    args = ap.parse_args()

    params = {"$limit": args.limit}
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(r.content)

    # also save csv for convenience
    df = pd.read_json(args.out)
    csv_path = Path(args.out).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print("saved", args.out, "and", csv_path, "rows=", len(df))

if __name__ == "__main__":
    main()