import os
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Gridsearch.Main import Gridsearch


# This file has some functions that are used in multiple scripts

# Load the scores with the help of the gridsearch class
def get_score(X, labels):
    # Start a fake search
    search = Gridsearch('kmeans', X)
    # Get the scores
    result = search.get_score(labels)
    return result


# Transform the scores to a dataframe
def get_score_df(cluster_df, X):
    """
    Loop over a dataframe with cluster labels
    cluster_df: A dataframe with different assignment of samples in each column
    X: The X data the columns are based of
    """
    # Start the loop over every cluster
    scores_df = {}
    for column in cluster_df:
        scores_df[column] = get_score(X, cluster_df[column].values)
    # Transform it to a dataframe with correct names and data types
    scores_df = pd.DataFrame(scores_df).T
    scores_df.index = scores_df.index.set_names(['n_clusters'])
    scores_df = scores_df.reset_index()
    scores_df['n_clusters'] = scores_df['n_clusters'].astype('int64')
    return scores_df


def merge_Results(directory, alg, norm_df):
    # Merge and calculate scores of a single set. This is mainly to put the hddc and hierachical clustering in
    # a silhouette score.
    grid_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    # Loop over the csv files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Load the hddc grid and clean it and save it again
            if 'Grid' in filename and alg in filename:
                df = pd.read_csv(f'{directory}/{filename}', index_col=0)
                df = df.T.reset_index(drop=True)
                grid_df = grid_df.append(df, ignore_index=True)
            # Load the clusters made by hddc or hierachical clustering
            if 'Clusters' in filename and alg in filename:
                try:
                    cluster_df[re.findall(r'\d+', filename)[0]] = pd.read_csv(f'{directory}/{filename}',
                                                                              index_col=0).values.ravel()
                except:
                    pass
    # Get all the silhouette scores for the made clusters
    scores_df = get_score_df(cluster_df, norm_df)
    # HDDC needs some additional cleaning compared to hierarchical
    if alg == 'HDDC':
        if 'comment' in grid_df.columns:
            grid_df = grid_df.replace('-Inf', np.nan)
            grid_df['comment'] = grid_df['comment'].replace(np.nan, 0)
            grid_df = grid_df.dropna()
        grid_df = grid_df.sort_values('BIC', ascending=False)
        grid_df['rank'] = np.arange(1, len(grid_df) + 1)
        grid_df = grid_df.drop('originalOrder', axis=1)
        return cluster_df, scores_df, grid_df
    else:
        return cluster_df, scores_df


# Function to load the different datassets
def load_sets(loaded_set):
    # Load the mapping of entrez ID to cancer type
    Entrez_Map = pd.read_csv('/data/g0017139/Set1/Entrezid_mapping_using_org_Hs_eg_db_03052021.txt',
                             sep=None, engine='python', header=0)
    Entrez_Map.sort_values(['CHR_Mapping', 'BP_Mapping'], axis=0, inplace=True)
    # Load the CCLE dataset
    if loaded_set == 1:
        # Gene expression data
        df = pd.read_csv(
            "/data/g0017139/Set1/CCLE__Affy_hgu133plus2_QCed_mRNA_NoDuplicates_CleanedIdentifiers_RMA-sketch_genelevel_using_jetscore.txt",
            sep=None, engine='python', header=0)
        df = df.T
        # Normalize
        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(df)
        # Cancer type
        Type_df = pd.read_csv('/data/g0017139/Set1/CCLE__Sample_To_TumorType.csv',
                              sep=None, engine='python', header=0)
        Type_df = Type_df.set_index('GSM_IDENTIFIER')
        # Gene_expression + Cancer type
        df = pd.concat([df, Type_df], axis=1)
    # Load the GPL570 data
    if loaded_set == 2:
        # Load the normal and non_normalized data
        df_normalized = pd.read_parquet("/data/g0017139/GPL570_norm.parquet")
        df = pd.read_parquet("/data/g0017139/GPL570_clean.parquet")
        # Load the cancer type
        Type_df = pd.read_csv("/data/g0017139/Set2/GPL570__Sample_To_TumorType.csv").set_index('GSM_IDENTIFIER')
        # Merge type with normal
        df = df.join(Type_df)
        # Clean them
        test = df['TYPE'].str.split(' - ', expand=True)[0]
        test = (['Leukemia' if ('leukemia' in x) else x for x in test])
        test = (['Lymphoma' if ('lymphoma' in x) else x for x in test])
        test = (['Sarcoma' if ('sarcoma' in x) or ('Sarcoma' in x) else x for x in test])
        df['TYPE'] = test
    # This was for a test isn't used anymore just here for backward compatibility
    if loaded_set == 3:
        df = pd.read_csv('/data/g0017139/Set2/Consensus mixing matrix.txt', sep='\t').set_index('Unnamed: 0')
        Type_df = pd.read_csv("/data/g0017139/Set2/GPL570__Sample_To_TumorType.csv").set_index('GSM_IDENTIFIER')
        # Don't normalize this one ??
        df_normalized = df.copy()
        df = df.join(Type_df)
    # Load the TCGA data
    if loaded_set == 4:
        # Load normalized and non-normalized
        df = pd.read_parquet("/data/g0017139/TCGA__RSEM.parquet")
        df_normalized = pd.read_parquet('/data/g0017139/TCGA__RSEM_norm.parquet')
        # Add type and merge them
        Type_df = pd.read_csv("/data/g0017139/Set2/TCGA_Sample_To_TumorType_20190920.csv").set_index('Name')
        df = df.join(Type_df)
    return df, df_normalized, Type_df
