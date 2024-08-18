import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set title and markdown
st.title('Data Visualization: Predicting Cancer Prognosis with Machine Learning')
st.markdown(
    """
    The following is the visualisation of data cluster of DLBCL patients who received R-CHOP regiment
    """
)

# Default dataset URL or path
default_csv_url = 'data LNH final.csv'

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv(default_csv_url, delimiter=';')

data = load_data()

# Drop specific columns
data = data.drop(['no', 'cluster', 'age2', 'nlr2', 'plr2', 'lmr2'], axis=1)

# Sample data setup (replace with actual data)
x_feat_cv = data.drop('outcome', axis=1)
y_pred = data['outcome']

# Standardize data
scaler = StandardScaler()
x_feat_scaled = pd.DataFrame(scaler.fit_transform(x_feat_cv), columns=x_feat_cv.columns)

# Combine scaled variables with outcome for PCA
cv = pd.concat([x_feat_scaled, pd.DataFrame({'outcome': y_pred})], axis=1)
cv['outcome'] = cv['outcome'].astype('category')

X = cv.drop(columns=['outcome'])
y = cv['outcome']

# PCA
@st.cache_data
def show_pca():
    pca = PCA(n_components=4)
    components = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_ * 100

    # Create DataFrame for components
    components_df = pd.DataFrame(components, columns=[f'PC{num+1}' for num in range(4)])
    components_df['outcome'] = y

    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=123)
    components_df['cluster'] = kmeans.fit_predict(components_df[['PC1', 'PC2', 'PC3', 'PC4']])
    silhouette_avg = silhouette_score(components_df[['PC1', 'PC2', 'PC3', 'PC4']], components_df['cluster'])
    #st.write(f'Silhouette Score: {silhouette_avg:.4f}')

    # Labeling
    components_df['label'] = np.where(components_df['cluster'] == 1, 'Survive', 'Not Survive')

    return components_df

components_df = show_pca()

# Define color mapping for clusters
color_map = {'Survive': 'rgba(0, 255, 255, 0.8)', 'Not Survive': 'rgba(255, 0, 0, 0.8)'}

@st.cache_data
def show_fig():
    # Plotting PCA components with clustering using Plotly
    fig = go.Figure()

    # Add 3D scatter plot
    fig.add_trace(go.Scatter3d(
        x=components_df['PC1'],
        y=components_df['PC2'],
        z=components_df['PC3'],
        mode='markers',
        marker=dict(
            size=4,
            color=components_df['label'].map(color_map).astype(str),  # Map labels to colors
            opacity=0.8,
            line=dict(
                width=2,
                color='black'
            )
        ),
        text=components_df['label'],
        name='Cluster'
    ))

    # Update layout
    fig.update_layout(
        title='PCA Components with K-means Clustering (3D)',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(50, 50, 50, 0.7)',
                gridcolor='rgba(255, 255, 255, 0.5)',
                gridwidth=2,
                showgrid=True,
                zerolinecolor='white',
                zerolinewidth=2,
                tickfont=dict(
                    color='white'
                )
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(50, 50, 50, 0.7)',
                gridcolor='rgba(255, 255, 255, 0.5)',
                gridwidth=2,
                showgrid=True,
                zerolinecolor='white',
                zerolinewidth=2,
                tickfont=dict(
                    color='white'
                )
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(50, 50, 50, 0.7)',
                gridcolor='rgba(255, 255, 255, 0.5)',
                gridwidth=2,
                showgrid=True,
                zerolinecolor='white',
                zerolinewidth=2,
                tickfont=dict(
                    color='white'
                )
            )
        ),
        paper_bgcolor='rgb(30, 30, 30)',
        plot_bgcolor='rgb(50, 50, 50)',
        legend=dict(
            title='Cluster Label',
            orientation='h',
            xanchor='center',
            x=0.5,
            y=-0.1,
            font=dict(
                color='white'
            )
        )
    )

    return fig

# Display the plot
fig = show_fig()
st.plotly_chart(fig)
