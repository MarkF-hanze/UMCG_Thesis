import faulthandler; faulthandler.enable()

import sys, getopt
import pickle
import os
import subprocess
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import warnings
import hdbscan
import umap
from tqdm import tqdm
import itertools
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

"""
Script to gridsearch with loads of different algorithms.
Did a search on lots of differnt algorithm not all made it to the paper
"""

# Basse gridsearch class. This one assumes a sklearn Object. If other thing are used overwrite according methods
class BaseAlg():
    def __init__(self):
        self.parameters = {}
        self.algorithm = None
        self.classes_ = None

    # Can be changed
    def set_classes(self, X):
        # Get the clasess of the fit algotroithm
        self.classes_ = self.algorithm.labels_

    def fit(self, X):
        # Fit the model and time how long it takes
        self.classes_ = None
        start_time = time.time()
        self.algorithm.fit(X)
        fit_time = time.time() - start_time
        # Set the classes of the fit model
        self.set_classes(X)
        return fit_time

    # Get the classes
    def get_classes(self):
        if self.classes_ is not None:
            return self.classes_
        else:
            warnings.warn('Classes have not been set')

    # Get the current gridsearch parameters
    def check_params(self, parameter):
        for name in parameter:
            if name not in self.algorithm.__dict__:
                warnings.warn(f'{name} not in algorithm')

    # Can be changed
    def helper_set(self, parameter):
        # Set the parameters of the algorithm
        self.algorithm.set_params(**parameter)

    # parameter should be a dictionary
    def set_params(self, parameter):
        self.check_params(parameter)
        self.helper_set(parameter)

    # Get the current parameters
    def get_params(self):
        return self.parameters

# Hierachical clustering need a change in the fit method for faster loading
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

# MAFIA is a tested algorithm that needed to overwrite everyting
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
        # Get the algorithms as save in a file
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
        # Clean it
        rows_to_cluster = pd.DataFrame(clusters, index=rows)
        rows_to_cluster = rows_to_cluster[~rows_to_cluster.index.duplicated(keep='last')]
        rows_to_cluster = rows_to_cluster.sort_index()
        self.classes_ = rows_to_cluster.values.ravel()

    # Fit it in a terminal (Doesn't work anymore because the algorithm file is removed)
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

# Kmeans is sklearn so doesn't need any changes
class Kmeans(BaseAlg):
    def __init__(self):
        self.parameters = {'n_clusters': np.arange(2, 31), 'batch_size': [128, 256, 512, 1024]}
        self.algorithm = MiniBatchKMeans(random_state=0)
        self.classes_ = None

# ESSC another custom algorithm need some small changes
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

# Elasticsspace subnet basicly sklearn object with some small changes
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

# UMAP + DBSCAN work the same as sklearn so only needs some small layout adjustments
class UHDBSCAN(BaseAlg):
    def __init__(self):
        self.parameters = {'DimReduction__n_neighbors': [25, 50],
                           'DimReduction__min_dist': [0.1, 0.75],
                           'DimReduction__n_components': [50, 10, 2],
                           'Clustering__min_cluster_size': [25, 100],
                           'Clustering__min_samples': [25, 100],
                           'Clustering__cluster_selection_epsilon': [0.1, 0.75],
                         #  'Clustering__cluster_selection_method': ['eom']
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

# Gridsearch with a BaseAlg object
class Gridsearch():
    def __init__(self, name, X):
        self.name = name
        self.X = X
        self.scores = []
        self.algorithm = self.get_alg(name)
        self.parameters = self.algorithm.get_params()

    # Get the name of the file to save it correcly with the name being the parameters used for this run
    def get_str(self,alg):
        the_str = str(alg)
        the_str = the_str.replace('(', '_')
        the_str = the_str.replace(')', '')
        the_str = the_str.replace('=', '__')
        the_str = the_str.replace(',', '_')
        the_str = the_str.replace('\n', '')
        for x in range(30):
            the_str = the_str.replace(' ', '')
        the_str = the_str.replace('[', '')
        the_str = the_str.replace(']', '')
        the_str = the_str.replace("'", '')
        the_str = the_str.replace("DimReduction", '')
        the_str = the_str.replace("Clustering", '')
        the_str = the_str.replace("Pipeline_steps", '')
        the_str = the_str.replace("dens_frac__0.0_dens_lambda__0.0", '')
        the_str = the_str.replace("____", '__')
        the_str = the_str.replace("__UMAP_", 'UMAP_')
        #the_str = the_str.replace(".", 'c')
        return the_str
        
    # Fit alg
    def fit(self, parameter, folder):
        # Fit it
        self.algorithm.set_params(parameter)
        fit_time = self.algorithm.fit(self.X)
        # Save alg
        algorithm_name = self.get_str(self.algorithm.algorithm)
        if 'UMAP' not in algorithm_name:
            with open(f'/scratch/g0017139/Models/{folder}/{algorithm_name}.pkl', 'wb') as f:
                pickle.dump(self.algorithm.algorithm, f, pickle.HIGHEST_PROTOCOL) 
        # Get the scores
        current_scores = self.get_score(self.algorithm.get_classes())
        current_scores['Fit_Time'] = fit_time
        current_scores.update(parameter)
        # If n_clusters doesn't exist make a custom entry with the amount of clusters
        if 'n_clusters' not in current_scores:
            current_scores['n_clusters'] = len(set(self.algorithm.get_classes()))
            if -1 in self.algorithm.get_classes():
                current_scores['n_clusters'] = current_scores['n_clusters'] - 1
              
        return current_scores

    # Check every combination given and calculate the score
    def start(self, folder):
        combinations = list(self.product_dict(**self.parameters))
        self.scores = []
        for combination in tqdm(combinations):
            self.scores.append(self.fit(combination, folder))
        return self.scores          
        
    # The current supported algorithms (Based on baseAlg implemented above)
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

    # Calculate a correlation and euclidean score based on the labels
    def get_score(self, labels):
        result = {}
        try:
            result['silhouette_score_euclidean'] = silhouette_score(self.X, labels, metric='euclidean', n_jobs=-1)
        except:
            result['silhouette_score_euclidean'] = np.nan
        try:
            result['silhouette_score_correlation'] = silhouette_score(self.X, labels, metric='correlation', n_jobs=-1)
        except:
            result['silhouette_score_correlation'] = np.nan
        return result

    # Get all possible combinations of variables in a dictionairy
    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

if __name__ == "__main__":
    # Set some of pythons limitations
    sys.setrecursionlimit(50000)
    argv = sys.argv[1:]
    # Read the parameters in the command line
    try:
        opts, args = getopt.getopt(argv, "f:a:s:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        # File to read
        if opt == "-f":
            input_path = arg
        # algorithm to use
        elif opt == "-a":
            algorithm_name = arg
        # File to save
        elif opt == "-s":
            folder = arg
    # Read input file and start gridsearch on this dataset with the given algorithm
    df = pd.read_csv(input_path, sep=None, engine='python', header=None)
    search = Gridsearch(algorithm_name, df)
    result = search.start(folder)
    # Save results
    with open(f'/home/g0017139/UMCG_Thesis/Scripts/Results/{folder}/{algorithm_name}{time.time()}.pkl', 'wb') as f:
            pickle.dump(pd.DataFrame(result), f, pickle.HIGHEST_PROTOCOL)
