import argparse
import pandas as pd
import joblib
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.feats.build import build_feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/processed/area_table.csv")
    ap.add_argument("--out_dir", default="reports/models")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = build_feats(df)
    y = df["jam_label"].astype(int)
    X = df.drop(columns=["jam_label", "area"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        "logreg": LogisticRegression(max_iter=300),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, prob)
        acc = accuracy_score(y_test, pred)
        out_rows.append({"model": name, "auc": float(auc), "acc": float(acc)})
        joblib.dump(m, f"{args.out_dir}/{name}.joblib")

    pathlib.Path("reports/results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv("reports/results/leaderboard.csv", index=False)

if __name__ == "__main__":
    main()