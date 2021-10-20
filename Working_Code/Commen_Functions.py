import os
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def get_score(X, labels):
    result = {}
    try:
        result['silhouette_score'] = silhouette_score(X, labels, metric='manhattan', n_jobs=-1)
    except:
        result['silhouette_score'] = np.nan

    try:
        result['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    except:
        result['calinski_harabasz_score'] = np.nan
    return result


def get_score_df(df, norm_df):
    scores_df = {}
    for column in df:
        scores_df[column] = get_score(norm_df, df[column].values)
    scores_df = pd.DataFrame(scores_df).T
    scores_df.index = scores_df.index.set_names(['n_clusters'])
    scores_df = scores_df.reset_index()
    scores_df['n_clusters'] = scores_df['n_clusters'].astype('int64')
    return scores_df


def merge_Results(directory, alg, norm_df):
    grid_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            if 'Grid' in filename and alg in filename:
                df = pd.read_csv(f'{directory}/{filename}', index_col=0)
                df = df.T.reset_index(drop=True)
                grid_df = grid_df.append(df, ignore_index=True)
            if 'Clusters' in filename and alg in filename:
                try:
                    cluster_df[re.findall(r'\d+', filename)[0]] = pd.read_csv(f'{directory}/{filename}',
                                                                              index_col=0).values.ravel()
                except:
                    pass
    scores_df = get_score_df(cluster_df, norm_df)

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


def load_sets(loaded_set):
    Entrez_Map = pd.read_csv('/data/g0017139/Set1/Entrezid_mapping_using_org_Hs_eg_db_03052021.txt',
                             sep=None, engine='python', header=0)
    Entrez_Map.sort_values(['CHR_Mapping', 'BP_Mapping'], axis=0, inplace=True)

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
        df = pd.concat([df, Type_df], axis=1)

    if loaded_set == 2:
        df_normalized = pd.read_parquet("/data/g0017139/GPL570_norm.parquet")
        df = pd.read_parquet("/data/g0017139/GPL570_clean.parquet")

        Type_df = pd.read_csv("/data/g0017139/Set2/GPL570__Sample_To_TumorType.csv").set_index('GSM_IDENTIFIER')
        df = df.join(Type_df)

    if loaded_set == 3:
        df = pd.read_csv('/data/g0017139/Set2/Consensus mixing matrix.txt', sep='\t').set_index('Unnamed: 0')
        Type_df = pd.read_csv("/data/g0017139/Set2/GPL570__Sample_To_TumorType.csv").set_index('GSM_IDENTIFIER')
        # Don't normalize this one ??
        df_normalized = df.copy()
        df = df.join(Type_df)

    if loaded_set == 4:
        df = pd.read_parquet("/data/g0017139/TCGA__RSEM.parquet")
        df_normalized = pd.read_parquet('/data/g0017139/TCGA__RSEM_norm.parquet')

        Type_df = pd.read_csv("/data/g0017139/Set2/TCGA_Sample_To_TumorType_20190920.csv").set_index('Name')
        df = df.join(Type_df)
    return df, df_normalized, Type_df


