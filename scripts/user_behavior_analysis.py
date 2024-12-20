# analysis/user_behavior_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

def aggregate_user_behaviour(df):
    """
    Groups the data by IMSI and aggregates behavior data for each user.
    """
    user_grouping = df.groupby(by="IMSI")
    user_behaviour = user_grouping.agg({
        "Bearer Id" : "count",
        "Dur. (ms)": "sum",
        "Total DL (Bytes)": "sum",
        "Total UL (Bytes)": "sum",
        "HTTP DL (Bytes)": "sum",
        "HTTP UL (Bytes)": "sum",
        "Social Media DL (Bytes)": "sum",
        "Social Media UL (Bytes)": "sum",
        "Google DL (Bytes)": "sum",
        "Google UL (Bytes)": "sum",
        "Email DL (Bytes)": "sum",
        "Email UL (Bytes)": "sum",
        "Youtube DL (Bytes)": "sum",
        "Youtube UL (Bytes)": "sum",
        "Netflix DL (Bytes)": "sum",
        "Netflix UL (Bytes)": "sum",
        "Gaming DL (Bytes)": "sum",
        "Gaming UL (Bytes)": "sum",
        "Other DL (Bytes)": "sum",
        "Other UL (Bytes)": "sum"
    })
    
    # Rename columns
    user_behaviour = user_behaviour.rename(columns={
        "Bearer Id" : "num xDr Sessions",
        "Dur. (ms)" : "Session Duration",
        "Total DL (Bytes)": "Total DL",
        "Total UL (Bytes)": "Total UL",
        "HTTP DL (Bytes)": "Volume DL",
        "HTTP UL (Bytes)": "Volume UL",
    })
    
    # Add total volume by summing upload and download volumes
    user_behaviour["Volume Total"] = user_behaviour["Volume DL"] + user_behaviour["Volume UL"]
    
    return user_behaviour

def get_skewness(df):
    """
    Calculates skewness for each column in the user behavior dataset.
    """
    skewness = df.apply(lambda x: skew(x.dropna()), axis=0)
    return skewness

def get_kurtosis(df):
    """
    Calculates kurtosis for each column in the user behavior dataset.
    """
    kurt = df.apply(lambda x: kurtosis(x.dropna()), axis=0)
    return kurt

def plot_box_plots(df, num_cols=3):
    """
    Creates box plots for each column in the user behavior dataset.
    """
    num_rows = math.ceil(len(df.columns) / num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for idx, col_name in enumerate(df.columns):
        sns.boxplot(data=df[col_name].astype(float), ax=axes[idx], orient='v', color="skyblue")
        axes[idx].set_title(col_name)
    
    plt.tight_layout()

def plot_violin_plots(df, num_cols=3):
    """
    Creates violin plots for each column in the user behavior dataset.
    """
    num_rows = math.ceil(len(df.columns) / num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    for idx, col_name in enumerate(df.columns):
        sns.violinplot(data=df[col_name].astype(float), ax=axes[idx], orient='v', color="lightgreen")
        axes[idx].set_title(col_name)
    
    plt.tight_layout()

def plot_correlation_matrix(df):
    """
    Plots the correlation matrix of the user behavior dataset.
    """
    correlation_matrix = df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = sns.diverging_palette(240, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, cmap=cmap, annot=True, ax=ax, linewidths=0.5)
    plt.tight_layout()

def perform_pca(df, n_components=4):
    """
    Performs PCA on the user behavior dataset and returns the principal components.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    return pca_df, pca.explained_variance_ratio_

def perform_pca_multiple_components(df, n_components=12):
    """
    Performs PCA with multiple components and returns the principal components.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    return pca_df, pca.explained_variance_ratio_
