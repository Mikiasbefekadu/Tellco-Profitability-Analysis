# user_satisfaction.py

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# Function to compute Euclidean distance for engagement or experience scores
def compute_euclidean_scores(data, cluster_centers, engagement_or_experience='engagement'):
    """
    Calculate the Euclidean distance score for engagement or experience.
    - data: DataFrame containing the user data.
    - cluster_centers: Array of cluster centers.
    - engagement_or_experience: 'engagement' or 'experience' to calculate respective scores.
    """
    if engagement_or_experience == 'engagement':
        cluster = cluster_centers[0]  # Assuming cluster 0 is the less engaged cluster
    elif engagement_or_experience == 'experience':
        cluster = cluster_centers[1]  # Assuming cluster 1 is the worst experience cluster
    else:
        raise ValueError("Invalid value for engagement_or_experience. Choose 'engagement' or 'experience'.")

    # Calculate Euclidean distance between user data points and the cluster center
    user_data_points = data[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
                             'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
                             'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].values
    distances = cdist(user_data_points, np.array([cluster]), metric='euclidean')
    return distances


# Function to compute satisfaction score (average of engagement and experience scores)
def compute_satisfaction_score(data, engagement_scores, experience_scores):
    # Assign scores to the DataFrame
    data['engagement_score'] = engagement_scores
    data['experience_score'] = experience_scores
    data['satisfaction_score'] = (data['engagement_score'] + data['experience_score']) / 2

    # Sort by satisfaction score and return the top 10 satisfied customers
    top_10_satisfied = data[['Bearer Id', 'satisfaction_score']].sort_values(by='satisfaction_score', ascending=False).head(10)
    return top_10_satisfied


# Function to build a regression model for predicting satisfaction score
def build_regression_model(data):
    """
    Build a regression model to predict satisfaction score.
    - data: DataFrame containing user data and satisfaction score.
    """
    # Feature columns and target
    X = data[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
              'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
              'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]
    y = data['satisfaction_score']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    return model


# Function to run K-Means clustering on engagement and experience scores
def build_regression_model  (data, n_clusters=2):
    """
    Run k-means clustering on engagement and experience scores.
    - data: DataFrame containing user data with engagement and experience scores.
    """
    # Prepare the data with engagement and experience scores
    kmeans_data = data[['engagement_score', 'experience_score']]
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(kmeans_data)
    
    return data, kmeans


# Function to aggregate satisfaction & experience scores per cluster
def aggregate_scores_per_cluster(data):
    """
    Aggregate the average satisfaction and experience scores per cluster.
    - data: DataFrame containing user data and cluster labels.
    """
    aggregated_scores = data.groupby('cluster').agg({
        'satisfaction_score': 'mean',
        'experience_score': 'mean'
    }).reset_index()
    
    return aggregated_scores


# Function to export data to MySQL database
def export_to_mysql(data, db_url, table_name="user_scores"):
    """
    Export the final table containing user engagement, experience, and satisfaction scores to MySQL.
    - data: DataFrame containing user data to be exported.
    - db_url: MySQL database connection URL.
    - table_name: The name of the table in the database.
    """
    # Create a connection to the MySQL database
    engine = create_engine(db_url)
    
    # Export the data to MySQL
    data.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"Data has been exported to {db_url} in the {table_name} table.")
