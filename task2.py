import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Load customer data
customer_data = pd.read_csv('/content/Mall_Customers (1).csv')
# Handle missing values using forward fill method
customer_data.fillna(method='ffill', inplace=True)
# Select only the numerical columns for scaling
numerical_features = ['Annual Income (k$)', 'Spending Score (1-100)']
numerical_data = customer_data[numerical_features]

# Standardize the numerical data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)
# Determine the optimal number of clusters using the elbow method
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia_values)
plt.title('Elbow Method for Optimal Cluster Selection')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
# Apply K-means clustering with the optimal number of clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
# Add cluster labels to the original data
customer_data['Cluster'] = cluster_labels
# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
# Interpret clusters by calculating the mean of each feature for each cluster
# Exclude non-numeric columns like 'Gender'
cluster_summary = customer_data.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)
