import os
import sys
import pandas as pd
import streamlit as st

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)
    

influence_metrics_final = pd.read_csv(r"https://raw.githubusercontent.com/john-adeojo/twitter-influence/main/data/02_intermediate/influence_metrics_final.csv")

# from ..scripts.clustering.cluster_streamlit import ClusterAnalysis


# import streamlit as st

import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class ClusterAnalysis:
    def __init__(self, dataframe, n_neighbors=15, min_cluster_size=5, min_dist=0.1, metric='euclidean'):
        self.dataframe = dataframe.copy()
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.min_dist = min_dist
        self.metric = metric

    def perform_umap(self):
        reducer = UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, metric=self.metric, random_state=42)
        umap_data = reducer.fit_transform(self.dataframe[['favorite_count_pf_mean', 'retweet_count_pf_mean', 'quote_count_pf_mean','reply_count_pf_mean', 'anger',	'joy',	'optimism',	'sadness',	'negative',	'neutral',	'positive']])
        self.dataframe['x'] = umap_data[:, 0]
        self.dataframe['y'] = umap_data[:, 1]

    def perform_hdbscan(self):
        np.random.seed(42)
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, metric=self.metric)
        self.dataframe['cluster'] = clusterer.fit_predict(self.dataframe[['x', 'y']])
    
    def plot_scatter(self):
        unique_clusters = sorted(self.dataframe['cluster'].unique())
        fig = go.Figure()

        for cluster in unique_clusters:
            cluster_data = self.dataframe[self.dataframe['cluster'] == cluster]
            fig.add_trace(go.Scatter(x=cluster_data['x'], y=cluster_data['y'], mode='markers', name='Cluster ' + str(cluster),
                                     marker=dict(size=6, opacity=0.4), hovertext=cluster_data['name'],
                                     text=cluster_data['name'], textposition='top center', textfont=dict(size=10, color='black')))

        fig.update_layout(title='Influence Clusters', showlegend=True, width=750, height=750)
        return fig
        # st.plotly_chart(fig, use_container_width=True)

    def run(self):
        self.perform_umap()
        self.perform_hdbscan()
        self.plot_scatter()



# Create input widgets in the sidebar
st.sidebar.header('UMAP and HDBSCAN Parameters')
n_neighbors = st.sidebar.slider('Number of Neighbors', 2, 50, 5)
min_cluster_size = st.sidebar.slider('Minimum Cluster Size', 2, 50, 5)
min_dist = st.sidebar.slider('Minimum Distance', 0.01, 1.0, 0.09, step=0.01)
metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'cosine', 'l1', 'l2'])

# Create an instance of ClusterAnalysis with the user-defined parameters
ca = ClusterAnalysis(influence_metrics_final, n_neighbors=n_neighbors, min_cluster_size=min_cluster_size, min_dist=min_dist, metric=metric)

# Run the analysis and display the scatter plot
ca.run()
scatter_fig = ca.plot_scatter()  # Get the figure object
st.plotly_chart(scatter_fig, use_container_width=True)