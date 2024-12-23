# user_engagement_analysis.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def aggregate_metrics(data):
    """
    Aggregate session metrics by MSISDN (Customer ID).
    
    """
    data['SessionFrequency'] = data.groupby('MSISDN/Number')['Bearer Id'].transform('count')
    data['SessionDuration'] = data['Dur. (ms)']
    data['TotalTraffic'] = data['Total UL (Bytes)'] + data['Total DL (Bytes)']
    aggregated_data = data.groupby('MSISDN/Number').agg(
        session_frequency=('SessionFrequency', 'sum'),
        session_duration=('SessionDuration', 'sum'),
        total_traffic=('TotalTraffic', 'sum')
    ).reset_index()
    return aggregated_data

def top_n_customers(data, column, n=10):
    """
    Return the top N customers based on a specific metric.
    """
    return data.sort_values(by=column, ascending=False).head(n)

def normalize_data(data, columns):
    """
    Normalize selected columns using StandardScaler.
    """
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data, scaler

def run_kmeans(data, columns, k=3):
    """
    Run K-means clustering and return the labels.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[columns])
    return data, kmeans

def elbow_method(data, columns, max_k=10):
    """
    Perform elbow method to find the optimal k.
    """
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[columns])
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method to Determine Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.grid()
    plt.show()

def compute_cluster_metrics(data, columns):
    """
    Compute min, max, mean, and total for each cluster.
    """
    cluster_summary = data.groupby('Cluster')[columns].agg(['min', 'max', 'mean', 'sum']).reset_index()
    return cluster_summary

def aggregate_traffic_per_application(data):
    """
    Aggregate total traffic per application per user.
    """
    app_traffic = data.groupby(['Application', 'MSISDN']).agg(
        total_traffic=('TotalTraffic', 'sum')
    ).reset_index()
    return app_traffic


def aggregate_traffic_per_application(data, top_n=3):
    """
    Aggregate total traffic per application per user and plot the top N applications with highest traffic.
    """
    # Define columns for each application
    app_columns = {
        'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
        'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
        'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
        'Youtube': ['Youtube DL (Bytes)', 'Youtube UL (Bytes)'],
        'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
        'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
    }

    # Aggregate traffic for each application
    app_traffic_list = []
    for app, cols in app_columns.items():
        app_data = data[['MSISDN/Number'] + cols].copy()
        app_data['Application'] = app
        app_data['TotalTraffic'] = app_data[cols].sum(axis=1)
        app_traffic_list.append(app_data[['MSISDN/Number', 'Application', 'TotalTraffic']])

    # Concatenate all applications
    app_traffic = pd.concat(app_traffic_list, ignore_index=True)

    # Aggregate traffic per application per user
    aggregated_traffic = app_traffic.groupby(['Application', 'MSISDN/Number']).agg(
        total_traffic=('TotalTraffic', 'sum')
    ).reset_index()

    # Aggregate total traffic per application
    total_traffic_per_app = aggregated_traffic.groupby('Application')['total_traffic'].sum().reset_index()

    # Sort applications by total traffic in descending order and select top N
    top_apps = total_traffic_per_app.sort_values('total_traffic', ascending=False).head(top_n)

    # Plot the top N applications by total traffic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='total_traffic', y='Application', data=top_apps, palette='viridis')
    plt.title(f"Top {top_n} Applications by Total Traffic", fontsize=16)
    plt.xlabel("Total Traffic (Bytes)", fontsize=14)
    plt.ylabel("Application", fontsize=14)
    plt.tight_layout()
    plt.show()

    return aggregated_traffic

# Example usage
# aggregated_traffic = aggregate_traffic_per_application(data, top_n=3)
