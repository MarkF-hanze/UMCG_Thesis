from IPython.display import SVG, display
from tqdm import tqdm
from pyclustertend import hopkins,ivat
from tqdm import tqdm


import hdbscan
import umap
import pickle
import os
import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from distinctipy import distinctipy
import holoviews as hv
from holoviews import opts, dim


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Commen_Funtions import get_score, get_score_df, merge_Results, load_sets




from bokeh.transform import factor_cmap, factor_mark
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, Dropdown, Select, Panel, Tabs, CustomJS
from bokeh.palettes import Category20
from bokeh.layouts import row, column
from bokeh.transform import cumsum

import panel as pn

def make_countplot(df, x, hue, pallete):
    pallette = sns.set_palette(pallete)
    # Make the figure
    fig = plt.figure(figsize=(10, 8))
    df = df.sort_values(hue)
    g = sns.countplot(x=x, hue=hue, palette=pallette, edgecolor=".6",
                      data=df)
    g.get_legend().remove()
    g.set_title('Absoulte counts')

    fig2 = plt.figure(figsize=(12, 8))
    counts = (df.groupby([hue])[x]
              .value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values(hue))
    g1 = sns.barplot(x=x, y="percentage", hue=hue,
                     data=counts, palette=pallette, edgecolor=".6")

    g1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    g1.set_title('Relative counts')
    return fig, fig2


def cluster_plot_bokeh(grid):
    if 'n_clusters' in grid.columns:
        df = grid.groupby(['n_clusters']).max().reset_index()
        df = df.sort_values('n_clusters')
        p = figure(width=700, height=500, x_axis_label='Number clusters', y_axis_label='Silhouette score')
        p.line(df['n_clusters'], df['silhouette_score'])

        p1 = figure(width=700, height=500, x_axis_label='Number clusters', y_axis_label='Calinski Harabasz score')
        p1.line(df['n_clusters'], df['calinski_harabasz_score'])
    else:
        df = grid.groupby(['K']).max().reset_index()
        TOOLTIPS = [
            ("Clusters", "@k"),
            ("Score", "@BIC"),
        ]
        p = figure(width=700, height=500, x_axis_label='Number clusters', y_axis_label='BIC score', tooltips=TOOLTIPS)
        p.line(df['K'], df['BIC'])

        p1 = figure(width=700, height=500, x_axis_label='Number clusters', y_axis_label='ICL score')
        p1.line(df['K'], df['ICL'])
    return p, p1


def do_dim_red(X):
    # Train PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X)
    # Train UMAP
    um = umap.UMAP(n_components=2, n_neighbors=90, min_dist=0.1, random_state=42)
    umap_components = um.fit_transform(X)

    # Put it in a dataframe
    df = pd.DataFrame(pca_components, columns=['PCAComponent1', 'PCAComponent2'])
    df['UMAPComponent1'] = umap_components[:, 0]
    df['UMAPComponent2'] = umap_components[:, 1]

    return df


def make_colors(df):
    cluster_colors = distinctipy.get_colors(max(15, len(set(df['Type']))),
                                colorblind_type='Deuteranomaly', n_attempts=10_000)
    cluster_colors = ['#%02x%02x%02x' % tuple((np.array(x)  * 250).astype(int)) for x in cluster_colors]

    color_mapper = dict(zip(set(df['Type']), cluster_colors))
    palette = []
    for key in sorted(color_mapper):
        palette.append(color_mapper[key])
    palette = sns.color_palette(palette)
    return cluster_colors, color_mapper, palette

def transform_clusters(clusters):
    clusters = clusters- 1
    clusters.columns = [int(x) for x in clusters.columns]
    clusters = clusters.reindex(sorted(clusters.columns), axis=1)
    return clusters


def get_hierarch(load_set, X):
    clusters, results =  merge_Results(f'/home/g0017139/UMCG_Thesis/Working_Code/Results/Set{load_set}/', 'Hierarch', X)
    clusters = transform_clusters(clusters)
    return clusters, results


def get_hddc(load_set, X):
    # THE HDDC
    clusters, results, grid =  merge_Results(f'/home/g0017139/UMCG_Thesis/Working_Code/Results/Set{load_set}/', 'HDDC', X)
    clusters = transform_clusters(clusters)
    return clusters, results, grid


def transform_results(results):
    results = results[results.groupby(['n_clusters'])['silhouette_score'].transform(max) == results['silhouette_score']]
    results = results.groupby('n_clusters').first()
    results = results.reset_index()
    return results

def get_kmeans(load_set, X):
    # Get the results of k-means for set 1
    with open(f"/home/g0017139/UMCG_Thesis/Working_Code/Results/Set{load_set}/kmeans.pkl", 'rb') as f:
            results = pickle.load(f)

    results = transform_results(results)

    clusters = pd.DataFrame()
    for index, row in results.iterrows():
        if row['n_clusters'] < 15:
            kmeans = MiniBatchKMeans(n_clusters=int(row['n_clusters']),
                                     random_state=0,
                                     batch_size=int(row['batch_size']))
            kmeans.fit(X)
            clusters[int(row['n_clusters'])] = kmeans.labels_
    return clusters, results

def get_dbscan(load_set, X):
    # Get the results of DBSCAN for set 1
    with open(f"/home/g0017139/UMCG_Thesis/Working_Code/Results/Set{load_set}/dbscan.pkl", 'rb') as f:
            results = pickle.load(f)

    results = transform_results(results)

    clusters = pd.DataFrame()
    for index, row in results.iterrows():
        if row['n_clusters'] < 15:
            pipe = Pipeline([('DimReduction',
                      umap.UMAP(
                          n_neighbors=int(row['DimReduction__n_neighbors']),
                          min_dist=float(row['DimReduction__min_dist']),
                          n_components=int(row['DimReduction__n_components']),
                          random_state=1)),
                     ('Clustering',
                     hdbscan.HDBSCAN(min_cluster_size=int(row['Clustering__min_cluster_size']),
                                    min_samples=int(row['Clustering__min_samples']),
                                    cluster_selection_epsilon=float(row['Clustering__cluster_selection_epsilon']),
                                    cluster_selection_method=row['Clustering__cluster_selection_method']))
                    ])
            pipe.fit(X)
            clusters[int(row['n_clusters'])] = pipe['Clustering'].labels_
    return clusters, results

def tab1(source, df):
    colors_or = distinctipy.get_colors(len(set(df['Type'])), colorblind_type='Deuteranomaly',
                                n_attempts=10_000)
    colors = ['#%02x%02x%02x' % tuple((np.array(x)  * 250).astype(int)) for x in colors_or]
    cmap = factor_cmap('Type', colors, list(set(df['Type'])))

    p = figure(width=550, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='PCA')
    p.scatter("PCAComponent1", "PCAComponent2", source=source, legend_field="Type",
              color = cmap
              )
    p.legend.visible=False
    p1 = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='UMAP')
    p1.scatter("UMAPComponent1", "UMAPComponent2", source=source, legend_field="Type",
                color = cmap
              )
    p1.add_layout(p1.legend[0], 'right')
    tab1 = pn.Row(p,p1)
    return tab1

def make_pca_umap(source, alg):
    # PCA UMAP Plot
    p1 = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='PCA')
    r1 = p1.scatter("PCAComponent1", "PCAComponent2", source=source, fill_color=alg,
               line_color=alg)

    p2 = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='UMAP')
    r2 = p2.scatter("UMAPComponent1", "UMAPComponent2", source=source, fill_color=alg,
               line_color=alg)
    return p1, p2


def sankey_plot(df):
    l1 = []
    l2 = []
    l3 = []
    for i in range(len(df.columns) - 1):
        for x in set(df.iloc[:,i]):
            for y in set(df.iloc[:,i + 1]):
                l1.append(f'{i}_{x}')
                l2.append(f'{i+1}_{y}')
                l3.append(len(df[((df.iloc[:,i] == x) & (df.iloc[:,i + 1] == y))]))
    sankey_df = pd.DataFrame(
        {'Group1': l1,
         'Group2': l2,
         'Count': l3
        })
    sankey_df = sankey_df[sankey_df['Count'] !=0]
    sankey = hv.Sankey(sankey_df.values)
    sankey.opts(opts.Sankey(label_position='right', width=1800, height=800))
    return sankey


def pie_plot(df, color_mapper):
    tabs_pie = []
    for column in df:
        if column != 'Type':
            figures = []
            for group, df_loop in df.groupby(column):
                size = len(df_loop)
                df_loop = df_loop.groupby('Type').count()[column].reset_index()
                df_loop['angle'] = df_loop[column]/df_loop[column].sum() * 2*np.pi
                df_loop['color'] = [color_mapper[x] for x in df_loop['Type'].values]
                df_loop.columns = [str(x) for x in df_loop.columns]
                p = figure(width=200, height=150, toolbar_location=None, match_aspect=True,
                           tools="hover", tooltips=f"@Type: @{column}", x_range=(-0.5, 1.0),
                           title=f'size: {size}')
                r = p.wedge(x=0, y=1, radius=0.4,
                        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                        line_color="white", fill_color='color', source=ColumnDataSource(df_loop))
                p.axis.axis_label=None
                p.axis.visible=False
                p.grid.grid_line_color = None
                figures.append(pn.pane.Bokeh(p))
            # Make the grid
            total = []
            count = len(figures)
            while True:
                if len(figures) > 4 :
                    total.append(pn.Row(*figures[0:4]))
                    figures = figures[4:]
                else:
                    total.append(pn.Row(*figures))
                    break
            tabs_pie.append((f'Clusters {column}', pn.Column(*total)))
    pie_p = pn.Tabs(*tabs_pie)
    return pie_p




def set_board(X, X_with_type, current_set):
    # Train PCA and UMAP
    df = do_dim_red(X)
    # TODO KLOPT DIT VOOR ALLE ANDERE DENK HET NIET
    df['Type'] = X_with_type['TYPE'].values
    df = df.fillna('UNKNOWN')
    # Clustering
    cluster_colors, color_mapper, palette = make_colors(df)

    # Get the results of hierarhical clustering for set 1
    results_grid = {}
    clusters = {}

    # Hierachical Clustering
    clusters['HierarchicalClustering'], results_grid['HierarchicalClustering'] = get_hierarch(current_set, X)
    df['HierarchicalClustering'] = [cluster_colors[x] for x in clusters['HierarchicalClustering'][2]]
    X_with_type['HierarchicalClustering'] = clusters['HierarchicalClustering'][2].values

    # KMEANS
    clusters['Kmeans'], results_grid['Kmeans'] = get_kmeans(current_set, X)
    df['Kmeans'] = [cluster_colors[x] for x in clusters['Kmeans'][2]]
    X_with_type['Kmeans'] = clusters['Kmeans'][2].values

    # DBSCAN
    clusters['DBSCAN'], results_grid['DBSCAN'] = get_dbscan(current_set, X)
    df['DBSCAN'] = [cluster_colors[x] for x in clusters['DBSCAN'][2]]
    X_with_type['DBSCAN'] = clusters['DBSCAN'][2].values

    # HDDC
    clusters['HDDC'], results_grid['HDDC'], grid = get_hddc(current_set, X)
    df['HDDC'] = [cluster_colors[x] for x in clusters['HDDC'][2]]
    X_with_type['HDDC'] = clusters['HDDC'][2].values

    # Everyting in a source
    source = ColumnDataSource(data=df)

    # Make the first tab
    explore_tab = tab1(source, df)

    # Make the alg tabs
    algs = ["HierarchicalClustering", "Kmeans", "DBSCAN", "HDDC"]
    tabs_cluster = []
    for alg in algs:
        clusters[alg]['Type'] = df['Type'].values
        # PCA UMAP Plot
        p2, p3 = make_pca_umap(source, alg)
        # HISTOGRAM
        p4, p5 = make_countplot(clusters[alg], 2, 'Type', palette)
        p4 = pn.pane.Matplotlib(p4, tight=True)
        p5 = pn.pane.Matplotlib(p5, tight=True)
        # LinePLOT
        p6, p7 = cluster_plot_bokeh(results_grid[alg])
        # SANKEY PLOT
        sankey = sankey_plot(clusters[alg])
        # PIE CHARTS
        pie_p = pie_plot(clusters[alg], color_mapper)
        # Make the tab
        tab = pn.Column(pn.Row(p2, p3), pn.Row(p4, p5), pn.Row(p6, p7), sankey, pie_p)
        tabs_cluster.append((alg, tab))
    return pn.Tabs(('Data Exporation', explore_tab), *tabs_cluster)

if __name__ == '__main__':
    tabs = []
    for LOADED_SET in range(1, 5):
        df, df_normalized, Type_df = load_sets(LOADED_SET)
        tabs.append(set_board(df_normalized, df, LOADED_SET))

    pn.Tabs(*tabs).save('Dashboard.html')

