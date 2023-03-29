import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px

class ClusterAnalysis:
    def __init__(self, dataframe, n_neighbors=15, min_cluster_size=5, min_dist=0.1):
        self.dataframe = dataframe.copy()
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_dist = min_dist

    def perform_umap(self):
        reducer = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        umap_data = reducer.fit_transform(self.dataframe[['favorite_count_pf_mean', 'retweet_count_pf_mean', 'quote_count_pf_mean','reply_count_pf_mean']])
        self.dataframe['x'] = umap_data[:, 0]
        self.dataframe['y'] = umap_data[:, 1]

    def perform_hdbscan(self):
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size)
        self.dataframe['cluster'] = clusterer.fit_predict(self.dataframe[['x', 'y']])

    # def plot_scatter(self):
    #     fig = px.scatter(self.dataframe, x='x', y='y', color='cluster', hover_name=self.dataframe['name'], opacity=0.4)
    #     fig.show()
    
    def plot_scatter(self):
        unique_clusters = sorted(self.dataframe['cluster'].unique())
        fig = px.scatter(self.dataframe, x='x', y='y', color='cluster', hover_name=self.dataframe['name'], opacity=0.4,
                         category_orders={"cluster": unique_clusters})
        fig.show()

    def run(self):
        self.perform_umap()
        self.perform_hdbscan()
        self.plot_scatter()
