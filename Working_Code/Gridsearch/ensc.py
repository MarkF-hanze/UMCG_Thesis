import pickle
import pandas as pd
import numpy as np


from gridsearch import gridsearch
normalized = pd.read_csv(
    "/home/g0017139/UMCG_Thesis/Working_Code/Results/gene_expression_norm.dat",
    sep=None, engine='python', header=None,
)

from subspaceClustering.cluster.selfrepresentation import ElasticNetSubspaceClustering
ensc = ElasticNetSubspaceClustering()
parameters = {'n_clusters': np.arange(2, 15, 1),
#              'affinity ': ['symmetrize', 'nearest_neighbors'],
              'tau': np.linspace(0.1, 1, 4),
              'gamma': [5, 50, 100, 500]
             }

ensc = ElasticNetSubspaceClustering(algorithm ='spams', n_jobs=-1)
df = pd.DataFrame(gridsearch(ensc, normalized, parameters))
with open('ensc2.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)