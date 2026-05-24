import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

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
# Correlation visualization (scatter plots)
# -------------------------------

plt.figure(figsize=(12,4))

# Spending vs Quantity
plt.subplot(1,3,1)
plt.scatter(customer_data['TotalSpending'], customer_data['TotalQuantity'])
plt.xlabel('TotalSpending')
plt.ylabel('TotalQuantity')
plt.title('Spending vs Quantity')

# Spending vs Frequency
plt.subplot(1,3,2)
plt.scatter(customer_data['TotalSpending'], customer_data['PurchaseFrequency'])
plt.xlabel('TotalSpending')
plt.ylabel('PurchaseFrequency')
plt.title('Spending vs Frequency')

# Quantity vs Frequency
plt.subplot(1,3,3)
plt.scatter(customer_data['TotalQuantity'], customer_data['PurchaseFrequency'])
plt.xlabel('TotalQuantity')
plt.ylabel('PurchaseFrequency')
plt.title('Quantity vs Frequency')

plt.tight_layout()
plt.show()


cor = customer_data.corr()
print(cor)

X_corr = customer_data[['TotalSpending', 'TotalQuantity']]
scaler =StandardScaler()
X_scaled = scaler.fit_transform(X_corr)

pca =PCA(n_components=1)

pc1 =pca.fit_transform(X_scaled)

customer_data['Spending_Quantity_PC1'] = pc1

# Keep other independent feature
final_data = customer_data[
    ['Spending_Quantity_PC1', 'PurchaseFrequency']
]

print(final_data.head())

cor = final_data.corr()
print(cor)
# -------------------------------
# 5. Features
# -------------------------------
X = final_data[['Spending_Quantity_PC1', 'PurchaseFrequency']]

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
plt.scatter(final_data['Spending_Quantity_PC1'],
            final_data['PurchaseFrequency'],
            c=customer_data['Cluster'])

plt.xlabel('PC1')
plt.ylabel('PurchaseFrequency')
plt.title('Clusters After PCA')

plt.show()

# # -------------------------------
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
# New customer
new_customer = pd.DataFrame({
    'TotalSpending': [5000],
    'TotalQuantity': [200],
    'PurchaseFrequency': [5]
})

# Step 1: Take correlated features
new_corr = new_customer[
    ['TotalSpending', 'TotalQuantity']
]

# Step 2: Apply SAME scaler
new_scaled = scaler.transform(new_corr)

# Step 3: Apply SAME PCA
new_pc1 = pca.transform(new_scaled)

# Step 4: Final input
new_final = pd.DataFrame({
    'Spending_Quantity_PC1': new_pc1.flatten(),
    'PurchaseFrequency': new_customer['PurchaseFrequency']
})

# Step 5: Predict
cluster = full_pipeline.predict(new_final)

print(cluster)

from sklearn.metrics import silhouette_score

score = silhouette_score(X_processed, customer_data['Cluster'])
print("Silhouette Score:", score)