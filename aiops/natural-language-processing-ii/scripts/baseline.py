#!/usr/bin/env python3
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def show_top_features(pipe, n=6):
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    for i, cls in enumerate(clf.classes_):
        top = names[np.argsort(clf.coef_[i])[::-1][:n]]
        print(f"{cls}: " + " | ".join(top))


def main():
    train = pd.read_csv("data/tickets_train.csv")
    val = pd.read_csv("data/tickets_val.csv")
    test = pd.read_csv("data/tickets_test.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])
    pipe.fit(train.text, train.label)

    val_pred = pipe.predict(val.text)
    test_pred = pipe.predict(test.text)
    metrics = {
        "model": "tfidf-logreg",
        "val_f1_macro": round(f1_score(val.label, val_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(test.label, test_pred), 4),
        "f1_macro": round(f1_score(test.label, test_pred, average="macro"), 4),
    }
    print(classification_report(test.label, test_pred, digits=3))
    print("== confusion matrix (rows=true, cols=pred) ==")
    print(pd.crosstab(test.label, test_pred, rownames=["true"], colnames=["pred"]))
    print("\n== top features per class ==")
    show_top_features(pipe)

    joblib.dump(pipe, "models/baseline.joblib")
    with open("reports/baseline_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("\n", metrics)


if __name__ == "__main__":
    main()
