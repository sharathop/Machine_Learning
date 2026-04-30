import math
from collections import Counter

# -----------------------------
# Step 1: Training dataset
# -----------------------------
# Format: [weight, sweetness]
X_train = [
    [150, 7],
    [170, 6],
    [140, 8],
    [130, 4],
    [120, 3],
    [110, 5]
]

# Labels: A = Apple, O = Orange
y_train = ['A', 'A', 'A', 'O', 'O', 'O']

# -----------------------------
# Step 2: Distance function
# -----------------------------
def euclidean_distance(p1, p2):
    sum_sq = 0
    for i in range(len(p1)):
        sum_sq += (p1[i] - p2[i]) ** 2
    return math.sqrt(sum_sq)

# -----------------------------
# Step 3: KNN function
# -----------------------------
def knn_predict(X_train, y_train, x_new, k=3):
    distances = []

    # Compute distance from new point to all training points
    for i in range(len(X_train)):
        dist = euclidean_distance(x_new, X_train[i])
        distances.append((dist, y_train[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Pick top K neighbors
    k_neighbors = distances[:k]

    # Extract labels of neighbors
    labels = [label for _, label in k_neighbors]

    # Majority vote
    most_common = Counter(labels).most_common(1)

    return most_common[0][0]

# -----------------------------
# Step 4: Test with new point
# -----------------------------
x_new = [145, 6]

prediction = knn_predict(X_train, y_train, x_new, k=3)

print("Prediction:", prediction)