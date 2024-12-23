# user_experience.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def aggregate_user_experience(data):
    # List of relevant parameters to aggregate
    params = [
        'TCP DL Retrans. Vol (Bytes)', 
        'TCP UL Retrans. Vol (Bytes)', 
        'Avg RTT DL (ms)', 
        'Avg RTT UL (ms)', 
        'Avg Bearer TP DL (kbps)', 
        'Avg Bearer TP UL (kbps)'
    ]
    
    # Handle missing and outlier values for each parameter
    for param in params:
        mean_val = data[param].mean()  # Compute mean value
        data[param] = data[param].fillna(mean_val)  # Replace NaN with mean
        
        # Calculate IQR to handle outliers
        Q1 = data[param].quantile(0.25)
        Q3 = data[param].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with the mean
        data[param] = np.where(
            (data[param] < lower_bound) | (data[param] > upper_bound),
            mean_val,
            data[param]
        )
    
    # Create a combined column for total TCP retransmissions
    data['TCP Retransmissions'] = data['TCP DL Retrans. Vol (Bytes)'] + data['TCP UL Retrans. Vol (Bytes)']
    
    return data


def compute_top_bottom_frequent(data):
    # Using the appropriate columns from the dataset
    tcp_dl_values = data['TCP DL Retrans. Vol (Bytes)']
    tcp_ul_values = data['TCP UL Retrans. Vol (Bytes)']
    rtt_dl_values = data['Avg RTT DL (ms)']
    rtt_ul_values = data['Avg RTT UL (ms)']
    tp_dl_values = data['Avg Bearer TP DL (kbps)']
    tp_ul_values = data['Avg Bearer TP UL (kbps)']

    # Function to compute top, bottom, and most frequent values
    def compute_stats(values, label):
        return {
            f"top_10_{label}": values.nlargest(10),
            f"bottom_10_{label}": values.nsmallest(10),
            f"most_frequent_{label}": values.mode().iloc[0] if not values.mode().empty else None,
        }

    # Compute for each metric
    results = {
        **compute_stats(tcp_dl_values, "tcp_dl"),
        **compute_stats(tcp_ul_values, "tcp_ul"),
        **compute_stats(rtt_dl_values, "rtt_dl"),
        **compute_stats(rtt_ul_values, "rtt_ul"),
        **compute_stats(tp_dl_values, "throughput_dl"),
        **compute_stats(tp_ul_values, "throughput_ul"),
    }

    return results

def compute_other_statistics(data):
    # Correct columns for statistics
    relevant_columns = [
        'TCP DL Retrans. Vol (Bytes)', 
        'TCP UL Retrans. Vol (Bytes)', 
        'Avg RTT DL (ms)', 
        'Avg RTT UL (ms)', 
        'Avg Bearer TP DL (kbps)', 
        'Avg Bearer TP UL (kbps)'
    ]
    
    # Ensure all columns exist in the dataset
    existing_columns = [col for col in relevant_columns if col in data.columns]
    
    if not existing_columns:
        raise ValueError("None of the required columns are present in the dataset.")
    
    # Compute mean, std, and median
    stats = data[existing_columns].agg(['mean', 'std', 'median'])
    return stats


def plot_throughput_per_handset(data):
    # Group by handset type and calculate the average throughput
    avg_throughput_dl = data.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean().sort_values(ascending=False)
    avg_throughput_ul = data.groupby('Handset Type')['Avg Bearer TP UL (kbps)'].mean().sort_values(ascending=False)

    # Plot average throughput for DL and UL per handset type
    plt.figure(figsize=(10, 6))
    avg_throughput_dl.plot(kind='bar', color='blue', alpha=0.7, label='Average DL Throughput (kbps)')
    avg_throughput_ul.plot(kind='bar', color='orange', alpha=0.7, label='Average UL Throughput (kbps)')
    plt.title('Average Throughput per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average Throughput (kbps)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_tcp_retransmissions_per_handset(data):
    # Group by handset type and calculate the average TCP retransmissions
    avg_tcp_dl = data.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False)
    avg_tcp_ul = data.groupby('Handset Type')['TCP UL Retrans. Vol (Bytes)'].mean().sort_values(ascending=False)

    # Plot average TCP retransmissions for DL and UL per handset type
    plt.figure(figsize=(10, 6))
    avg_tcp_dl.plot(kind='bar', color='blue', alpha=0.7, label='Average DL TCP Retransmissions (Bytes)')
    avg_tcp_ul.plot(kind='bar', color='orange', alpha=0.7, label='Average UL TCP Retransmissions (Bytes)')
    plt.title('Average TCP Retransmissions per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average TCP Retransmissions (Bytes)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def perform_kmeans_clustering(data, k=3):
    # Select relevant features for clustering
    features = [
        'TCP DL Retrans. Vol (Bytes)', 
        'TCP UL Retrans. Vol (Bytes)', 
        'Avg RTT DL (ms)', 
        'Avg RTT UL (ms)', 
        'Avg Bearer TP DL (kbps)', 
        'Avg Bearer TP UL (kbps)'
    ]
    
    # Ensure the data contains the required features
    data_for_clustering = data[features].dropna()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    data_for_clustering['Cluster'] = kmeans.fit_predict(data_for_clustering)

    # Plot the clustering result
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_for_clustering, x='Avg Bearer TP DL (kbps)', y='Avg Bearer TP UL (kbps)', hue='Cluster', palette='viridis')
    plt.title('K-Means Clustering of Users by Experience')
    plt.xlabel('Average Bearer TP DL (kbps)')
    plt.ylabel('Average Bearer TP UL (kbps)')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # Describe the clusters
    cluster_centers = kmeans.cluster_centers_
    cluster_descriptions = {
        f"Cluster {i+1}": {
            "Average TCP DL Retransmissions": cluster_centers[i][0],
            "Average TCP UL Retransmissions": cluster_centers[i][1],
            "Average RTT DL": cluster_centers[i][2],
            "Average RTT UL": cluster_centers[i][3],
            "Average Throughput DL": cluster_centers[i][4],
            "Average Throughput UL": cluster_centers[i][5]
        }
        for i in range(k)
    }

    return cluster_descriptions
