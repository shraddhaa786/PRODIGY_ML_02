# PRODIGY_ML_02


# Customer Segmentation Using K-Means Clustering

## Project Overview

This aims to group customers of a retail store into distinct segments based on their purchasing behaviors using K-means clustering. By identifying these customer segments, businesses can tailor their marketing strategies and enhance customer satisfaction.

### Dataset

The dataset used for this project is available on Kaggle: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

- **Features**:
  - `CustomerID`: Unique ID of the customer (removed during processing).
  - `Gender`: Gender of the customer (removed as it's categorical and not required for clustering).
  - `Age`: Age of the customer.
  - `Annual Income (k$)`: Customer’s annual income in thousands.
  - `Spending Score (1-100)`: A score assigned by the mall based on customer spending habits.

## Project Structure

- `Mall_Customers.csv`: The dataset file containing customer information.
- `kmeans_clustering.py`: Python script for performing K-means clustering on the dataset.
- `README.md`: This file, providing an overview of the project.

## Approach

### 1. Data Preprocessing:
- The dataset was loaded using `pandas` and inspected for missing values.
- Irrelevant columns (`CustomerID` and `Gender`) were removed as they don’t contribute to the clustering process.
- Missing values (if any) in the numerical columns were filled with the column mean.
- The features used for clustering were `Annual Income (k$)` and `Spending Score (1-100)`.

### 2. Feature Scaling:
- Since K-means clustering is sensitive to the scale of data, the selected features were standardized using `StandardScaler`.

### 3. K-Means Clustering:
- K-means clustering was applied with 5 clusters (chosen arbitrarily) to group customers based on their income and spending score.
- The optimal number of clusters can also be determined using the elbow method (not covered in this script).

### 4. Visualization:
- A scatter plot was generated to visualize the customer segments, with each cluster colored differently.

## Installation & Usage

### Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

### Running the Code

1. Clone this repository and download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
   
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the clustering script:
   ```bash
   python kmeans_clustering.py
   ```

## Results

The K-means algorithm divided the customers into 5 distinct segments based on their annual income and spending score. The generated scatter plot visualizes these clusters, helping businesses understand different customer groups.

## Optional Enhancements

- The optimal number of clusters can be determined using the elbow method or silhouette score.
- More features, like `Age`, could be included in the clustering for deeper insights.

## Conclusion

This demonstrates the application of K-means clustering for customer segmentation. By understanding distinct customer groups, businesses can make data-driven decisions and improve their marketing strategies.

