import hdbscan
import umap
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from distinctipy import distinctipy
import holoviews as hv
from holoviews import opts, dim


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from Commen_Functions import get_score, get_score_df, merge_Results, load_sets
from Gridsearch.Main import Gridsearch



from bokeh.transform import factor_cmap, factor_mark
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, Dropdown, Select, Panel, Tabs, CustomJS, LinearAxis, Range1d, HoverTool
from bokeh.transform import cumsum
from bokeh.io import export_png

import panel as pn
"""
Create a html dashboard with all the clustering results
"""
def make_countplot(df, x, hue, pallete):
    # Make a countplot for every cluster type
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
    export_png(fig, filename=f"/data/g0017139/Images/{LOADED_SET}_counplot1.png")
    export_png(fig2, filename=f"/data/g0017139/Images/{LOADED_SET}_counplot2.png")
    return fig, fig2


def cluster_plot_bokeh(grid, alg):
    # Create a plot how the scores change in regards to the amount of clusters made
    if 'n_clusters' in grid.columns:
        df = grid.groupby(['n_clusters']).max().reset_index()
        df = df.sort_values('n_clusters')
        source = ColumnDataSource(df)
        p = figure(width=1400, height=500, x_axis_label='Number clusters', y_axis_label='Correlation score')
      
        l2 = p.line('n_clusters', 'silhouette_score_euclidean', source=source,
                    legend_label='silhouette_score_euclidean', color='red')
        p.add_tools(HoverTool(renderers=[l2], tooltips=[('Silhoette euclidian',"@silhouette_score_euclidean{0.00}")],
                              mode='vline'))        
        l3 = p.line('n_clusters', 'silhouette_score_correlation', source=source,
                    legend_label='silhouette_score_correlation', color='hotpink')
        p.add_tools(HoverTool(renderers=[l3], tooltips=[('Silhoette correlation',"@silhouette_score_correlation{0.00}")],
                              mode='vline'))
   
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
    export_png(p, filename=f"/data/g0017139/Images/{LOADED_SET}_{alg}_clusterScores.png")
    return p

# Do PCA and UMAP on the data
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

# Create distinct colors for every cancer type in df
def make_colors(df):
    cluster_colors = distinctipy.get_colors(max(15, len(set(df['Type']))),
                                colorblind_type='Deuteranomaly', n_attempts=10_000)
    cluster_colors = ['#%02x%02x%02x' % tuple((np.array(x) * 250).astype(int)) for x in cluster_colors]

    color_mapper = dict(zip(set(df['Type']), cluster_colors))
    palette = []
    for key in sorted(color_mapper):
        palette.append(color_mapper[key])
    palette = sns.color_palette(palette)
    return cluster_colors, color_mapper, palette

# Do some cleaning on the clusters to make them all equal format
def transform_clusters(clusters):
    clusters = clusters - 1
    clusters.columns = [int(x) for x in clusters.columns]
    clusters = clusters.reindex(sorted(clusters.columns), axis=1)
    return clusters

# Load hierachical results
def get_hierarch(load_set, X):
    with open(f'/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{load_set}/Merged_results.pkl', 'rb') as f:
        results = pickle.load(f)['Hierarch']
    with open(f'/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{load_set}/MergedClustersHierarch.pkl', 'rb') as f:
        clusters = pickle.load(f)
    clusters = transform_clusters(clusters)
    return clusters, results

# Load hddc results
def get_hddc(load_set, X):
    # THE HDDC
    with open(f'/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{load_set}/Merged_results.pkl', 'rb') as f:
        results = pickle.load(f)['hddc']
    with open(f'/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{load_set}/MergedClustersHDDC.pkl', 'rb') as f:
        clusters = pickle.load(f)
    clusters = transform_clusters(clusters)
    return clusters, results 

# Only leave the max silhouette score
def transform_results(results):
    results = results[results.groupby(
        ['n_clusters'])['silhouette_score_euclidean'].transform(max) == results['silhouette_score_euclidean']]
    results = results.groupby('n_clusters').first()
    results = results.reset_index()
    return results

# Get the K-means results
def get_kmeans(load_set, X):
    # Get the results of k-means for set
    with open(f"/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{load_set}/kmeans.pkl", 'rb') as f:
            results = pickle.load(f)
    # Transform the results
    results = transform_results(results)
    clusters = pd.DataFrame()
    # Retrain the model for later use
    for index, row in results.iterrows():
        kmeans = MiniBatchKMeans(n_clusters=int(row['n_clusters']),
                                 random_state=0,
                                 batch_size=int(row['batch_size']))
        kmeans.fit(X)
        clusters[int(row['n_clusters'])] = kmeans.labels_
    return clusters, results

# Get the dbscan results
def get_dbscan(load_set, X):
    # Load them
    # Get the results of DBSCAN for set 1
    with open(f"/home/g0017139/UMCG_Thesis/Scripts/Results/TSet{load_set}/dbscan.pkl", 'rb') as f:
            results = pickle.load(f)
    # Retrain the model for later use
    results = transform_results(results)
    clusters = pd.DataFrame()
    for index, row in results.iterrows():
        f1 = float(row['Clustering__cluster_selection_epsilon'])
        if f1 == 1.0:
            f1 = int(f1)
        try:
            pipe = Pipeline([('DimReduction',
                          umap.UMAP(
                              n_neighbors=int(row['DimReduction__n_neighbors']),
                              min_dist=float(row['DimReduction__min_dist']),
                              n_components=int(row['DimReduction__n_components']))),
                         ('Clustering',
                         hdbscan.HDBSCAN(min_cluster_size=int(row['Clustering__min_cluster_size']),
                                        min_samples=int(row['Clustering__min_samples']),
                                        cluster_selection_epsilon=f1,
                                        cluster_selection_method=row['Clustering__cluster_selection_method'],
                                        core_dist_n_jobs=-1
                                        ))
                        ])
        except KeyError:
            pipe = Pipeline([('DimReduction',
                          umap.UMAP(
                              n_neighbors=int(row['DimReduction__n_neighbors']),
                              min_dist=float(row['DimReduction__min_dist']),
                              n_components=int(row['DimReduction__n_components']))),
                         ('Clustering',
                         hdbscan.HDBSCAN(min_cluster_size=int(row['Clustering__min_cluster_size']),
                                        min_samples=int(row['Clustering__min_samples']),
                                        cluster_selection_epsilon=f1,
                                        core_dist_n_jobs=-1
                                        ))
                         ])
        pipe.fit(X)
        clusters[int(row['n_clusters'])] = pipe['Clustering'].labels_
    return clusters, results

# Make the first step for data exploration
def tab1(source, df):
    # Get colors
    colors_or = distinctipy.get_colors(len(set(df['Type'])), colorblind_type='Deuteranomaly',
                                n_attempts=10_000)
    colors = ['#%02x%02x%02x' % tuple((np.array(x)  * 250).astype(int)) for x in colors_or]
    cmap = factor_cmap('Type', colors, list(set(df['Type'])))
    # Make scatterplots
    p = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='PCA')
    p.scatter("PCAComponent1", "PCAComponent2", source=source, legend_field="Type",
              color = cmap
              )  
    p.legend.visible=False
    p1 = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='UMAP')
    p1.scatter("UMAPComponent1", "UMAPComponent2", source=source, legend_field="Type",
                color = cmap
              )
    # Layout
    p1.add_layout(p1.legend[0], 'right')
    if LOADED_SET == 4:
      p1.legend.label_text_font_size = '3pt'
    tab1 = pn.Row(p, p1)
    export_png(p, filename=f"/data/g0017139/Images/{LOADED_SET}_PCAREAL.png")
    export_png(p1, filename=f"/data/g0017139/Images/{LOADED_SET}_UMAPREAL.png")
    return tab1

def make_pca_umap(source, colors):
    # PCA UMAP Plot for
    p1 = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='PCA')
    r1 = p1.scatter("PCAComponent1", "PCAComponent2", source=source, fill_color=colors,
               line_color=colors)

    p2 = figure(width=700, height=500, x_axis_label='Component 1', y_axis_label='Component 2', title='UMAP')
    r2 = p2.scatter("UMAPComponent1", "UMAPComponent2", source=source, fill_color=colors,
               line_color=colors)
    return p1, p2



# Plot the sankey, this plot shows how the cluster distribution changed
def sankey_plot(df, alg):
    # Drop type because it is not important
    df = df.drop('Type', axis=1)
    # Fill a dataframe to the correct holoviews format
    l1 = []
    nodes = []
    edges = []
    # Loop over every cluster type (amount of clusters)
    for i in range(len(df.columns) - 1):
        # Loop over every cluster
        for x in set(df.iloc[:,i]):
            # loop over every cluster of the next amount and draw the connections
            for y in set(df.iloc[:,i + 1]):
                if f'{i}_{x}' not in l1:
                    l1.append(f'{i}_{x}')
                    nodes.append((f'{i}_{x}', ''))
                if f'{i+1}_{y}' not in l1:
                    l1.append(f'{i+1}_{y}')
                    nodes.append((f'{i+1}_{y}', ''))
                if len(df[((df.iloc[:,i] == x) & (df.iloc[:,i + 1] == y))]) !=0:
                    edges.append((f'{i}_{x}', f'{i+1}_{y}',len(df[((df.iloc[:,i] == x) & (df.iloc[:,i + 1] == y))])))
    # Make the sankey plot
    nodes = hv.Dataset(nodes, 'index', 'label')
    sankey = hv.Sankey((edges, nodes), ['From', 'To'])
    sankey.opts(
    opts.Sankey(labels='label', label_position='right', width=1800, height=800, 
                edge_color='#058896', node_color='#058896', color_index = None))
    hv.save(sankey, f"/data/g0017139/Images/{LOADED_SET}_{alg}_sankey.png", fmt='png')
    return sankey

# Make the pieplots
def pie_plot(df, color_mapper, alg):
    tabs_pie = []
    # Loop over every plot type (different clustering options)
    for column in df:
        if column != 'Type':
            # Make a pieplot for every cluster
            figures = []
            for group, df_loop in df.groupby(column):
                # Make a single pieplot
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
                # Layout
                p.axis.axis_label=None
                p.axis.visible=False
                p.grid.grid_line_color = None
                figures.append(pn.pane.Bokeh(p))
                export_png(p, filename=f"/data/g0017139/Images/{LOADED_SET}_{alg}_pieplot_{column}_{group}.png")
            # Make the grid a square grid
            total = []
            while True:
                if len(figures) > 4 :
                    total.append(pn.Row(*figures[0:4]))
                    figures = figures[4:]
                else:
                    total.append(pn.Row(*figures))
                    break
            tabs_pie.append((f'Clusters {column}', pn.Column(*total)))
    # return the differnt pieplots tabs
    pie_p = pn.Tabs(*tabs_pie)
    return pie_p

# Make the clusterheamap (Shows the same information as the pieplots)
def heatmap(df, alg):
    tabs = []
    # Loop over every cluster type
    for column in df:
        if column != 'Type':
            start_df = df.copy()
            start_df['count'] = 1
            # Count how often every cluster type appears
            count_df = start_df.groupby([column, 'Type']).sum().reset_index()
            sum_df = start_df.groupby(['Type']).sum()
            new_col = []
            for index, row in count_df.iterrows():
                new_col.append(sum_df.loc[row['Type'],:]['count'])
            # Take the relative counts
            count_df['Percentage'] = count_df['count'] / new_col
            count_df = count_df.drop('count', axis=1)
            # Pivot it to make it correct input for seaborn
            data = pd.pivot_table(count_df, values='Percentage', index='Type', columns=column, fill_value=0)
            # Make the clustermap from seaborn "YlGnBu"
            fig = sns.clustermap(data, method="ward", col_cluster=False,  cmap="rocket", figsize=(10,20))
            fig.savefig(f"/data/g0017139/Images/{LOADED_SET}_{alg}_heatmap_{column}.png")
            fig = pn.pane.Matplotlib(fig.fig, tight=True)
            tabs.append((f'Clusters {column}', fig))
    # Return all the seperate tabs
    fig = pn.Tabs(*tabs)
    return fig
    
# Combine all the different plots in a single html file
def set_board(X, X_with_type, current_set):
    # Train PCA and UMAP
    df = do_dim_red(X)
    df['Type'] = X_with_type['TYPE'].values
    df = df.fillna('UNKNOWN')
    # Clustering
    cluster_colors, color_mapper, palette = make_colors(df)

    # Get the results of hierarhical clustering for set 1
    results_grid = {}
    clusters = {}
    numbers = {1:[2, 2, 2, 2],
               2:[11, 2, 11, 2],
               4:[26, 13, 26, 23]}
    # Hierachical Clustering
    clusters['HierarchicalClustering'], results_grid['HierarchicalClustering'] = get_hierarch(current_set, X)
    df['HierarchicalClustering'] = [cluster_colors[x] for x in clusters['HierarchicalClustering'][numbers[current_set][0]]]
    X_with_type['HierarchicalClustering'] = clusters['HierarchicalClustering'][numbers[current_set][0]].values

    # KMEANS
    clusters['Kmeans'], results_grid['Kmeans'] = get_kmeans(current_set, X)
    df['Kmeans'] = [cluster_colors[x] for x in clusters['Kmeans'][numbers[current_set][1]]]
    X_with_type['Kmeans'] = clusters['Kmeans'][numbers[current_set][1]].values

    # DBSCAN
    clusters['DBSCAN'], results_grid['DBSCAN'] = get_dbscan(current_set, X)
    df['DBSCAN'] = [cluster_colors[x] for x in clusters['DBSCAN'][numbers[current_set][2]]]
    X_with_type['DBSCAN'] = clusters['DBSCAN'][numbers[current_set][2]].values

    # HDDC
    clusters['HDDC'], results_grid['HDDC'] = get_hddc(current_set, X)
    df['HDDC'] = [cluster_colors[x] for x in clusters['HDDC'][numbers[current_set][3]]]
    X_with_type['HDDC'] = clusters['HDDC'][numbers[current_set][3]].values

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
        # LinePLOT
        p6 = cluster_plot_bokeh(results_grid[alg], alg)
        # SANKEY PLOT
        sankey = sankey_plot(clusters[alg], alg)
        # PIE CHARTS
        pie_p = pie_plot(clusters[alg], color_mapper, alg)
        # Heat map
        heatmap_fig = heatmap(clusters[alg], alg)
        # Make the tab
        tab = pn.Column(pn.Row(p2, p3), p6, sankey, heatmap_fig, pie_p)
        tabs_cluster.append((alg, tab))
    return pn.Tabs(('Data Exporation', explore_tab), *tabs_cluster)

if __name__ == '__main__':
    hv.extension('bokeh')
    tabs = []
    # Make a different dashboard file for every set (In one html file the loading time would become to high)
    for LOADED_SET in range(1, 5):
        if LOADED_SET != 3:
            df, df_normalized, Type_df = load_sets(LOADED_SET)
            set_board(df_normalized, df, LOADED_SET).save(f'/home/g0017139/UMCG_Thesis/Scripts/Results/Dashboard_{LOADED_SET}.html')
            plt.clf()
        

