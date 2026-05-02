import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# -------------------------------
# 1. Load data
# -------------------------------
data = pd.read_excel('Online Retail.xlsx')

# -------------------------------
# 2. Cleaning
# -------------------------------
data = data.dropna(subset=['CustomerID', 'Description'])
data = data[data['Quantity'] > 0]
data = data[data['UnitPrice'] > 0]

# -------------------------------
# 3. Feature Engineering
# -------------------------------
data['Total_price'] = data['Quantity'] * data['UnitPrice']

customer_data = data.groupby('CustomerID').agg({
    'Total_price': 'sum',
    'Quantity': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()

customer_data.columns = [
    'CustomerID',
    'TotalSpending',
    'TotalQuantity',
    'PurchaseFrequency'
]

# -------------------------------
# 4. Visualization (RAW DATA)
# -------------------------------
customer_data[['TotalSpending','TotalQuantity','PurchaseFrequency']].hist()
plt.suptitle("Raw Data Distribution")
plt.show()

plt.scatter(customer_data['TotalSpending'], customer_data['TotalQuantity'])
plt.xlabel('Spending')
plt.ylabel('Quantity')
plt.title('Raw Data Distribution')
plt.show()

# -------------------------------
# 5. Features
# -------------------------------
X = customer_data[['TotalSpending', 'TotalQuantity', 'PurchaseFrequency']]

# -------------------------------
# 6. Pipeline (log + scaling)
# -------------------------------
pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())
])

X_processed = pipeline.fit_transform(X)

# -------------------------------
# 7. Show transformed data
# -------------------------------
print("\n--- LOG TRANSFORMED DATA ---")
log_only = np.log1p(X)
log_only.hist()
plt.suptitle("Log Transformed Data")
plt.show()

print("\n--- SCALED DATA ---")
processed_df = pd.DataFrame(X_processed, columns=X.columns)
processed_df.hist()
plt.suptitle("Scaled Data")
plt.show()

# -------------------------------
# 8. Elbow Method
# -------------------------------
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# -------------------------------
# 9. Final Model 
# -------------------------------
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_processed)

# -------------------------------
# 10. Cluster Analysis
# -------------------------------
print("\n--- CLUSTER SUMMARY ---")
print(customer_data.groupby('Cluster').mean())

# -------------------------------
# 11. Cluster Visualization
# -------------------------------
plt.scatter(customer_data['TotalSpending'],
            customer_data['TotalQuantity'],
            c=customer_data['Cluster'])

plt.xlabel('Spending')
plt.ylabel('Quantity')
plt.title('Customer Clusters')
plt.show()

# -------------------------------
# 12. FULL PIPELINE (for prediction)
# -------------------------------
full_pipeline = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, n_init=10, random_state=42))
])

# Train full pipeline
full_pipeline.fit(X)

# -------------------------------
# 13. Predict NEW customer
# -------------------------------
new_customer = [[5000, 200, 5]]  


cluster = full_pipeline.predict(new_customer)

print("\nNew Customer belongs to Cluster:", cluster[0])

from sklearn.metrics import silhouette_score

score = silhouette_score(X_processed, customer_data['Cluster'])
print("Silhouette Score:", score)