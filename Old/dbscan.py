import hdbscan
import umap
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from gridsearch import gridsearch
normalized = pd.read_csv(
    "/home/g0017139/UMCG_Thesis/Working_Code/Results/gene_expression_norm.dat",
    sep=None, engine='python', header=None,
)

parameters = {'DimReduction__n_neighbors': np.arange(30,100,20),
              'DimReduction__min_dist': np.linspace(0,1,3),
              'DimReduction__n_components': np.arange(1,100,25)[::-1],
              'Clustering__min_cluster_size': [2 ,25 ,50 ,75 ,100],
              'Clustering__min_samples': [2 ,25 ,50 ,75 ,100],
              'Clustering__cluster_selection_epsilon': [0.1, 0.5, 1],
              'Clustering__cluster_selection_method': ['eom', 'leaf']
}
pipe = Pipeline([('DimReduction',
                  umap.UMAP(
                      n_neighbors=30,
                      min_dist=0.0,
                      n_components=2,
                      random_state=42)),
                 ('Clustering',
                 hdbscan.HDBSCAN())
                ])
df = pd.DataFrame(gridsearch(pipe, normalized, parameters))
with open('dbscan.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)