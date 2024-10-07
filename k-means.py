###import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#df = pd.read_csv(r'C:\Users\Shraddha\OneDrive\Prodigy\PRODIGY_ML_02\Mall_Customers.csv')
#numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#df.drop(columns=['CustomerID', 'Gender'], inplace=True, errors='ignore')
#X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
#kmeans = KMeans(n_clusters=5, random_state=42)
#df['Cluster'] = kmeans.fit_predict(X_scaled)
#plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='rainbow')
#plt.xlabel('Annual Income (k$)')
#plt.ylabel('Spending Score (1-100)')
#plt.title('Customer Segments')
#plt.show()
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\Shraddha\OneDrive\Prodigy\PRODIGY_ML_02\Mall_Customers.csv')

# Data exploration (optional: for understanding the dataset structure)
print(df.head())  # Print the first 5 rows
print(df.info())  # Overview of data types and missing values

# Data preprocessing
# Selecting only the numerical columns for clustering
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Filling missing values in numerical columns with the mean (if any)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Dropping irrelevant columns like 'CustomerID' and categorical columns like 'Gender'
df.drop(columns=['CustomerID', 'Gender'], inplace=True, errors='ignore')

# Selecting the features for clustering: Annual Income and Spending Score
# You can also add 'Age' as a feature for better clustering if you wish
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling the data (K-means is sensitive to scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
            c=df['Cluster'], cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.show()

# Optional: save the clustered data if needed
# df.to_csv('Clustered_Customers.csv', index=False)

# Optional: Inspect the final dataframe with clusters
print(df.head())  # See the first 5 rows with cluster labels
