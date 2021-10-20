import sys, getopt
import pickle
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import warnings
import time
from TSKFS.fuzzy_cluster import ESSC
from subspaceClustering.cluster.selfrepresentation import ElasticNetSubspaceClustering
from sklearn.pipeline import Pipeline
import hdbscan
import umap
from tqdm import tqdm
import itertools
import time
import numpy as np

from multiprocessing import Pool
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm.contrib.concurrent import process_map
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class BaseAlg():
    def __init__(self):
        self.parameters = {}
        self.algorithm = None
        self.classes_ = None

    # Can be changed
    def set_classes(self, X):
        self.classes_ = self.algorithm.labels_

    def fit(self, X):
        self.classes_ = None
        start_time = time.time()
        self.algorithm.fit(X)
        fit_time = time.time() - start_time
        self.set_classes(X)
        return fit_time

    def get_classes(self):
        if self.classes_ is not None:
            return self.classes_
        else:
            warnings.warn('Classes have not been set')

    def check_params(self, parameter):
        for name in parameter:
            if name not in self.algorithm.__dict__:
                warnings.warn(f'{name} not in algorithm')

    # Can be changed
    def helper_set(self, parameter):
        self.algorithm.set_params(**parameter)

    # parameter should be a dictionary
    def set_params(self, parameter):
        self.check_params(parameter)
        self.helper_set(parameter)

    def get_params(self):
        return self.parameters
    
class Hierachical(BaseAlg):
    def __init__(self, X):
        self.parameters = {'n_clusters': np.arange(2,25),
        }
        self.algorithm = AgglomerativeClustering()
        self.corr = 1 - np.absolute(pd.DataFrame(X).T.corr().values)
        
        
    def fit(self, X):
        self.classes_ = None
        start_time = time.time()
        self.algorithm.fit(self.corr)
        fit_time = time.time() - start_time
        self.set_classes(X)
        return fit_time


class MAFIA(BaseAlg):
    def __init__(self):
        self.parameters = {'a': np.linspace(0.5, 5, 5),
                           'b': np.linspace(0.25, 0.75, 4),
                           'n': np.arange(100, 2001, 250),
                           'u': np.arange(1, 11, 5),
                           'M': np.arange(15, 31, 5),
                          }
        self.algorithm = None
        self.classes_ = None
        self.current_params = None

    # Can be changed
    def set_classes(self, X):
        # Load all the classes 
        rows = []
        clusters = []
        directory = '/data/g0017139/MAFIA'
        for filename in os.listdir(directory):
            if filename.endswith(".idx"):
                loadedrow = pd.read_table(f"{directory}/{filename}",sep="  ", header=None, engine='python').values.tolist()
                os.remove(f"{directory}/{filename}")
                clusters.extend(np.repeat(int(filename.split('-')[1].replace('.idx','')), len(loadedrow)))
                rows.extend(loadedrow)
        rows = np.array(rows).ravel()
        clusters = np.array(clusters)
        # Get some outliers
        test = np.arange(0,len(X))
        for x in test:
            if x not in rows:
                rows = np.append(rows, x)
                clusters = np.append(clusters, -1)
        rows_to_cluster = pd.DataFrame(clusters, index=rows)
        rows_to_cluster = rows_to_cluster[~rows_to_cluster.index.duplicated(keep='last')]
        rows_to_cluster = rows_to_cluster.sort_index()
        self.classes_ = rows_to_cluster.values.ravel()

    def fit(self, X):
        self.classes_ = None
        pd.DataFrame(X).to_csv("/data/g0017139/MAFIA/X.dat", sep = " ",header=False, index=False)
        start_time = time.time()
        subprocess.run(f"/home/g0017139/UMCG_Thesis/Working_Code/bin/cppmafia /data/g0017139/MAFIA/X.dat -a {self.current_params['a']} -b {self.current_params['b']} -n {self.current_params['n']} -u {self.current_params['u']} -M {self.current_params['M']}",
                shell=True)
        fit_time = time.time() - start_time
        self.set_classes(X)
        return fit_time

    def get_classes(self):
        if self.classes_ is not None:
            return self.classes_
        else:
            warnings.warn('Classes have not been set')

    # Can be changed
    def helper_set(self, parameter):
        self.current_params = parameter

    # parameter should be a dictionary
    def set_params(self, parameter):
        self.helper_set(parameter)

    def get_params(self):
        return self.parameters


class Kmeans(BaseAlg):
    def __init__(self):
        self.parameters = {'n_clusters': np.arange(2, 16), 'batch_size': [128, 256, 512, 1024]}
        self.algorithm = MiniBatchKMeans(random_state=0)
        self.classes_ = None


class ESSCGrid(BaseAlg):
    def __init__(self):
        self.parameters = {'n_cluster': np.arange(2, 15, 1),
                           'eta': [0.01, 0.25, 0.5, 0.75, 1.0],
                           'gamma': [1, 5, 10, 50, 100, 500, 1000]
                           }
        self.algorithm = ESSC(None)
        self.classes_ = None

    def set_classes(self, X):
        self.classes_ = self.algorithm.predict(X)
        self.classes_ = [np.argmax(q) for q in self.classes_]

    # parameter should be a dictionary
    def helper_set(self, parameter):
        for name in parameter:
            setattr(self.algorithm, name, parameter[name])


class ENSC(BaseAlg):
    def __init__(self):
        self.parameters = {'n_clusters': np.arange(2, 6, 1),
                           'tau': np.linspace(0.1, 1, 3),
                           'gamma': [1, 5, 100, 500]
                           }
        self.algorithm = ElasticNetSubspaceClustering(algorithm='spams')
        self.classes_ = None

    def helper_set(self, parameter):
        for name in parameter:
            setattr(self.algorithm, name, parameter[name])


class UHDBSCAN(BaseAlg):
    def __init__(self):
        self.parameters = {'DimReduction__n_neighbors': [25, 50, 100],
                           'DimReduction__min_dist': [0.1, 0.5, 1],
                           'DimReduction__n_components': [50, 10, 2],
                           'Clustering__min_cluster_size': [25, 50, 100],
                           'Clustering__min_samples': [25, 50, 100],
                           'Clustering__cluster_selection_epsilon': [0.1, 0.5, 1],
                           'Clustering__cluster_selection_method': ['eom', 'leaf']
                           }
        for name in self.parameters:
            if isinstance(self.parameters[name], type(np.array)):
                self.parameters[name] = self.parameters[name].tolist()
        self.algorithm = Pipeline([('DimReduction',
                                    umap.UMAP()),
                                   ('Clustering',
                                    hdbscan.HDBSCAN(core_dist_n_jobs=-1))
                                   ])
        self.classes_ = None

    def set_classes(self, X):
        self.classes_ = self.algorithm['Clustering'].labels_

    def check_params(self, parameter):
        for name in parameter:
            if name not in self.algorithm.get_params():
                warnings.warn(f'{name} not in algorithm')

    def helper_set(self, parameter):
        self.algorithm.set_params(**parameter)


class Gridsearch():
    def __init__(self, name, X):
        self.name = name
        self.X = X
        self.scores = []
        self.algorithm = self.get_alg(name)
        self.parameters = self.algorithm.get_params()
        
    # Fit alg
    def fit(self, parameter):
        self.algorithm.set_params(parameter)
        fit_time = self.algorithm.fit(self.X)
        current_scores = self.get_score(self.algorithm.get_classes())
        current_scores['Fit_Time'] = fit_time
        current_scores.update(parameter)
        if 'n_clusters' not in current_scores:
            current_scores['n_clusters'] = len(set(self.algorithm.get_classes()))
            if -1 in self.algorithm.get_classes():
                current_scores['n_clusters'] = current_scores['n_clusters'] - 1
              
        return current_scores

    def start(self):
        MAX = 6
        combinations = list(self.product_dict(**self.parameters))
        print(f'cpu count: {os.cpu_count()}')
        print(f'used: {MAX}')
        self.scores = process_map(self.fit, combinations, max_workers=MAX, chunksize=10)
        return self.scores          
        

    def get_alg(self, name):
        if name == 'kmeans':
            class_alg = Kmeans()
        elif name == 'essc':
            class_alg = ESSCGrid()
        elif name == 'ensc':
            class_alg = ENSC()
        elif name == 'dbscan':
            class_alg = UHDBSCAN()
        elif name == 'mafia':
            class_alg = MAFIA()
        elif name == 'hierach':
            class_alg = Hierachical(self.X)
        else:
            raise ValueError(f'Algorithm {name} is not implemented chose "kmeans" "essc" "ensc" "dbscan" "MAFIA", "hierach"')
        return class_alg

    def get_score(self, labels):
        result = {}
        try:
            result['silhouette_score_euclidean'] = silhouette_score(self.X, labels, metric='euclidean', n_jobs=-1)
        except:
            result['silhouette_score'] = np.nan
        try:
            result['silhouette_score_jaccard'] = silhouette_score(self.X, labels, metric='jaccard', n_jobs=-1)
        except:
            result['silhouette_score'] = np.nan
        try:
            result['silhouette_score_correlation'] = silhouette_score(self.X, labels, metric='correlation', n_jobs=-1)
        except:
            result['silhouette_score'] = np.nan
        try:
            result['silhouette_score_mahalanobis'] = silhouette_score(self.X, labels, metric='mahalanobis', n_jobs=-1)
        except:
            result['silhouette_score'] = np.nan
        try:
            result['calinski_harabasz_score'] = calinski_harabasz_score(self.X, labels)
        except:
            result['calinski_harabasz_score'] = np.nan

        return result

    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "f:a:s:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-f":
            input_path = arg
        elif opt == "-a":
            algorithm_name = arg
        elif opt == "-s":
            folder = arg
    df = pd.read_csv(input_path, sep=None, engine='python', header=None)
    search = Gridsearch(algorithm_name, df)
    result = search.start()
    with open(f'/home/g0017139/UMCG_Thesis/Working_Code/Results/{folder}/{algorithm_name}{time.time()}.pkl', 'wb') as f:
            pickle.dump(pd.DataFrame(result), f, pickle.HIGHEST_PROTOCOL)
