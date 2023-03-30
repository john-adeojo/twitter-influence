import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
import plotly.graph_objects as go


class ClusterAnalysis:
    def __init__(self, dataframe, n_neighbors=15, min_cluster_size=5, min_dist=0.1):
        self.dataframe = dataframe.copy()
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_dist = min_dist

    def perform_umap(self):
        reducer = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, random_state=42)
        umap_data = reducer.fit_transform(self.dataframe[['favorite_count_pf_mean', 'retweet_count_pf_mean', 'quote_count_pf_mean','reply_count_pf_mean']])
        self.dataframe['x'] = umap_data[:, 0]
        self.dataframe['y'] = umap_data[:, 1]

    def perform_hdbscan(self):
        np.random.seed(42)
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size)
        self.dataframe['cluster'] = clusterer.fit_predict(self.dataframe[['x', 'y']])
        self.dataframe.to_csv(r"C:\Users\johna\anaconda3\envs\twitter-influence-env\twitter-influence\data\02_intermediate\tweet_analysis_data.csv")

    # def plot_scatter(self):
    #     fig = px.scatter(self.dataframe, x='x', y='y', color='cluster', hover_name=self.dataframe['name'], opacity=0.4)
    #     fig.show()
    
    def plot_scatter(self):
        unique_clusters = sorted(self.dataframe['cluster'].unique())
        fig = go.Figure()

        for cluster in unique_clusters:
            cluster_data = self.dataframe[self.dataframe['cluster'] == cluster]
            fig.add_trace(go.Scatter(x=cluster_data['x'], y=cluster_data['y'], mode='markers', name='Cluster ' + str(cluster),
                                     marker=dict(size=6, opacity=0.4), hovertext=cluster_data['name'],
                                     text=cluster_data['name'], textposition='top center', textfont=dict(size=10, color='black')))

        fig.update_layout(showlegend=True, width=500, height=500)
        fig.show()
# hovertext=cluster_data['name']

    def run(self):
        self.perform_umap()
        self.perform_hdbscan()
        self.plot_scatter()
