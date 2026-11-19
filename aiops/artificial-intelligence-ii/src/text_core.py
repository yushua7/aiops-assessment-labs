import numpy as np


def top_terms(pipeline, n=5):
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    names = vec.get_feature_names_out()
    return {str(cls): [str(names[i]) for i in np.argsort(clf.coef_[k])[::-1][:n]]
            for k, cls in enumerate(clf.classes_)}
