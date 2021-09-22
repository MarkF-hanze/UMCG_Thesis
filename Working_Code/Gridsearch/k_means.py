import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from gridsearch import gridsearch
normalized = pd.read_csv(
    "/home/g0017139/UMCG_Thesis/Working_Code/Results/gene_expression_norm.dat",
    sep=None, engine='python', header=None,
)


parameters = {'n_clusters': np.arange(2,30),
              'batch_size': [128, 256, 512, 1024]}
kmeans = MiniBatchKMeans(random_state=0)
df = pd.DataFrame(gridsearch(kmeans, normalized, parameters))
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)