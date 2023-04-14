import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/raw/nyc_speeds.csv")
    ap.add_argument("--out_csv", default="data/processed/area_table.csv")
    ap.add_argument("--min_rows", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    borough_col = None
    for c in ["borough", "boro", "boroname", "borough_name"]:
        if c in df.columns:
            borough_col = c
            break

    speed_col = None
    for c in ["speed", "speed_mph", "avg_speed", "travel_speed"]:
        if c in df.columns:
            speed_col = c
            break

    if borough_col is None:
        borough_col = "link_id" if "link_id" in df.columns else df.columns[0]

    if speed_col is None:
        for c in df.columns:
            if "speed" in c.lower():
                speed_col = c
                break

    if speed_col is None:
        raise RuntimeError("could not find speed column")

    df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce")
    df = df.dropna(subset=[speed_col])

    g = df.groupby(borough_col)[speed_col].agg(["count", "mean", "median", "std"]).reset_index()
    g = g.rename(columns={borough_col: "area", "mean": "avg_speed", "median": "med_speed", "std": "speed_std"})
    g = g[g["count"] >= args.min_rows].reset_index(drop=True)

    g["jam_label"] = (g["avg_speed"] < 10.0).astype(int)
    g.to_csv(args.out_csv, index=False)
    print("wrote", args.out_csv, "areas=", len(g))

if __name__ == "__main__":
    main()