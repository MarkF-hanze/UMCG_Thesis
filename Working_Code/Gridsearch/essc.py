import pickle
import pandas as pd
import numpy as np
from gridsearch import gridsearch
normalized = pd.read_csv(
    "/home/g0017139/UMCG_Thesis/Working_Code/Results/gene_expression_norm.dat",
    sep=None, engine='python', header=None,
)

from ..TSKFS.fuzzy_cluster import ESSC
parameters = {'n_cluster': np.arange(2,20, 1),
              'eta': [0, 0.1, 0.2, 0.3, 0.5, 0.7 , 0.9],
              'gamma': [1, 2, 5, 10, 50, 100, 1000]
             }
essc = ESSC(None)

df = pd.DataFrame(gridsearch(essc, normalized, parameters))
with open('essc.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)