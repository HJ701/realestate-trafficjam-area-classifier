import pandas as pd
import math

def build_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speed_cv"] = out["speed_std"] / (out["avg_speed"].abs() + 1e-6)
    out["log_count"] = (out["count"] + 1).apply(lambda x: math.log(x))
    return out