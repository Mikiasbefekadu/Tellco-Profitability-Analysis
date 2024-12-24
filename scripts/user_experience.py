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
    metrics = {
        'tcp_dl': data['TCP DL Retrans. Vol (Bytes)'],
        'tcp_ul': data['TCP UL Retrans. Vol (Bytes)'],
        'rtt_dl': data['Avg RTT DL (ms)'],
        'rtt_ul': data['Avg RTT UL (ms)'],
        'throughput_dl': data['Avg Bearer TP DL (kbps)'],
        'throughput_ul': data['Avg Bearer TP UL (kbps)']
    }

    def compute_stats(values):
        return {
            "top_10": values.nlargest(10),
            "bottom_10": values.nsmallest(10),
            "most_frequent": values.mode().iloc[0] if not values.mode().empty else None,
        }

    results = {}

    for label, values in metrics.items():
        stats = compute_stats(values)
        results[label] = stats

        # Visualization
        plt.figure(figsize=(12, 4))

        # Top 10 visualization
        plt.subplot(1, 3, 1)
        stats["top_10"].plot(kind="bar", color="skyblue")
        plt.title(f"Top 10 {label}")
        plt.xticks(rotation=45)

        # Bottom 10 visualization
        plt.subplot(1, 3, 2)
        stats["bottom_10"].plot(kind="bar", color="orange")
        plt.title(f"Bottom 10 {label}")
        plt.xticks(rotation=45)

        # Most frequent value visualization
        plt.subplot(1, 3, 3)
        plt.bar([label], [stats["most_frequent"]], color="green")
        plt.title(f"Most Frequent {label}")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    return results




def compute_satisfaction_score(data, engagement_scores, experience_scores):
    # Assign scores to the DataFrame
    data['engagement_score'] = engagement_scores
    data['experience_score'] = experience_scores
    data['satisfaction_score'] = (data['engagement_score'] + data['experience_score']) / 2

    # Sort by satisfaction score and return the top 10 satisfied customers
    top_10_satisfied = data[['Bearer Id', 'satisfaction_score']].sort_values(by='satisfaction_score', ascending=False).head(10)
    return top_10_satisfied



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
