import pandas as pd


def make_features(df: pd.DataFrame):
    v = df["value"]
    f = pd.DataFrame(index=df.index)
    f["value"] = v
    for k in (1, 3, 6, 12):
        f[f"lag_{k}"] = v.shift(k)
    for k in (1, 12):
        f[f"diff_{k}"] = v.diff(k)
    for w in (12, 36, 288):
        r = v.rolling(w)                      # trailing window only
        f[f"roll_mean_{w}"] = r.mean()
        f[f"roll_std_{w}"] = r.std()
    f["zscore_288"] = (v - f["roll_mean_288"]) / (f["roll_std_288"] + 1e-6)
    f["dev_from_mean_36"] = v - f["roll_mean_36"]
    f["hour"] = df.index.hour
    f["dayofweek"] = df.index.dayofweek
    ok = f.notna().all(axis=1).values
    X = f[ok].astype("float64")
    y = df["label"][ok]
    return X, y
