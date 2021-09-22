from tqdm import tqdm
import itertools
import time
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def get_score(X, labels):
    result = {}
    try:
        result['silhouette_score'] = silhouette_score(X, labels, metric='euclidean')
    except:
        result['silhouette_score'] = np.nan

    try:
        result['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    except:
        result['calinski_harabasz_score'] = np.nan

    try:
        result['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    except:
        result['davies_bouldin_score'] = np.nan
    return result


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def fit(alg, X, parameter, clustering):
    if isinstance(alg, Pipeline):
        alg.set_params(**parameter)
    else:
        for name in parameter:
            if isinstance(parameter[name], np.generic):
                setattr(alg, name, parameter[name].item())
            else:
                setattr(alg, name, parameter[name])
    start_time = time.time()
    alg.fit(X)
    fit_time = time.time() - start_time
    # Get scores and make dataframe
    try:
        if isinstance(alg, Pipeline):
            yp = alg['Clustering'].labels_
        else:
            yp = alg.labels_
    except (AttributeError, TypeError):
        yp = alg.predict(X)
        if clustering == 'Soft':
            yp = [np.argmax(q) for q in yp]
    current_scores = get_score(X, yp)
    current_scores['Fit_Time'] = fit_time
    current_scores.update(parameter)
    return current_scores


def gridsearch(alg, X, parameters, clustering='Hard'):
    scores = []
    for parameter in tqdm(product_dict(**parameters), total=len(list(product_dict(**parameters)))):
        # Fit alg
        current_scores = fit(alg, X, parameter, clustering)
        scores.append(current_scores)
    return scores

