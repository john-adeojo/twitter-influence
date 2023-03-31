import os
import sys
import streamlit as st
import pandas as pd


notebook_dir = os.path.dirname(os.path.abspath("__file__"))
project_dir = os.path.dirname(notebook_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)
    

# import data 
influence_metrics_final = pd.read_csv(r"https://raw.githubusercontent.com/john-adeojo/twitter-influence/main/data/02_intermediate/influence_metrics_final.csv")
tweets_with_sentiment = pd.read_csv(r"https://github.com/john-adeojo/twitter-influence/blob/main/data/02_intermediate/tweets_with_sentiment.csv?raw=true")


import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


# define cluster class
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
        

        
# define function to create a contingency table with counts of positive and non-positive samples for each cluster
from scipy.stats import chi2_contingency

def run_chisquare_analysis(df, var):
    contingency_table = pd.crosstab(df['cluster'], df[var])

    # Perform the Chi-square test
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)

    # Calculate the standardized residuals
    standardized_residuals = (contingency_table - ex) / np.sqrt(ex)
    
    #standardized_residuals.index.name = None
    standardized_residuals = standardized_residuals.reset_index()

    print("Chi2 Stat:", chi2_stat)
    print("P Value:", p_value)
    print("Degrees of Freedom:", dof)
    # print("Expected Frequency Table:")
    # print(ex)

    return standardized_residuals   


def heatmap(cats, title, xlabel, df):
    # Prepare the data for the heatmap
    heatmap_data = df.set_index('cluster')[cats]

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, linewidths=0.5, center=0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cluster')

    # Display the heatmap in Streamlit
    st.pyplot(fig)
        

# Streamlit dashboards 

# Create input widgets in the sidebar
st.title('Twitter Influence Clusters')
st.markdown("### Produced by [John Adeojo](https://www.john-adeojo.com/)")
st.sidebar.header('UMAP and HDBSCAN Parameters')
st.sidebar.text('Adjust hyperparameters and see \n the impact on influencer clustering')
n_neighbors = st.sidebar.slider('Number of Neighbors', 2, 50, 5)
min_cluster_size = st.sidebar.slider('Minimum Cluster Size', 2, 50, 5)
min_dist = st.sidebar.slider('Minimum Distance', 0.01, 1.0, 0.09, step=0.01)
metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'l1', 'l2'])

# Create an instance of ClusterAnalysis with the user-defined parameters
ca = ClusterAnalysis(influence_metrics_final, n_neighbors=n_neighbors, min_cluster_size=min_cluster_size, min_dist=min_dist, metric=metric)

# Run the analysis and display the scatter plot
ca.run()
scatter_fig = ca.plot_scatter()  # Get the figure object
st.plotly_chart(scatter_fig, use_container_width=True)

# create data frame for analysis
analysis_df = ca.dataframe
tweet_level_metrics = tweets_with_sentiment.merge(how='left', right=analysis_df[['user_id', 'cluster']], left_on='user_id', right_on='user_id')

# Add a section header for the Chi-square analysis and heatmap generation
st.header("Chi-Square Analysis and Heatmap")

# run analysis on emotion and sentiment 
standardized_residuals_emotion = run_chisquare_analysis(df=tweet_level_metrics, var='emotion')
standardized_residuals_sentiment = run_chisquare_analysis(df=tweet_level_metrics, var='sentiment')

# generate heatmaps for emotion and sentiment
heatmap(cats=['negative', 'neutral', 'positive'], title='Sentiment by Cluster', xlabel='Sentiment', df=standardized_residuals_sentiment)
heatmap(cats=['anger', 'joy', 'optimism', 'sadness'], title='Emotion by Cluster', xlabel='Emotion', df=standardized_residuals_emotion)

# print clusters 
clusters = list(analysis_df['cluster'].drop_duplicates().sort_values())

for cluster in clusters:
    st.write(f'cluster_{cluster}')
    names = analysis_df.loc[analysis_df['cluster']==cluster]['name']
    st.write(names)

