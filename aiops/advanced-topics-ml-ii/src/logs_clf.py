import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import ART, BASE

df = pd.read_csv(BASE / "data/loghub/BGL_2k.log_structured.csv")
y = (df.Label != "-").astype(int).values


def build_features(df, use_metadata):
    cv = CountVectorizer(token_pattern=r"[A-Za-z]{3,}", lowercase=True, min_df=2, binary=True)
    X_text = cv.fit_transform(df.Content.astype(str))
    names = [f"tok_{t}" for t in cv.get_feature_names_out()]
    X = X_text
    if use_metadata:
        meta = pd.get_dummies(df[["Level", "Component"]].astype(str))
        X = sparse.hstack([X_text, sparse.csr_matrix(meta.values.astype(np.float32))])
        names = names + list(meta.columns)
    return X.astype(np.float32).tocsr(), names


results = {}
for variant, use_meta in (("content+metadata", True), ("content_only", False)):
    X, names = build_features(df, use_meta)
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                             min_child_samples=5, n_jobs=2, random_state=42, verbose=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    res = {"pr_auc": average_precision_score(y, p), "f1": f1_score(y, (p >= 0.5).astype(int)),
           "n_features": X.shape[1]}
    clf.fit(X, y)
    gain = pd.Series(clf.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)
    res["top_features"] = gain.head(5).index.tolist()
    results[variant] = res
    print(f"{variant:17s} features={X.shape[1]:4d} PR-AUC={res['pr_auc']:.3f} F1={res['f1']:.3f}")
    print("   top by gain:", res["top_features"])

json.dump(results, open(ART / "metrics_logs.json", "w"), indent=2)
