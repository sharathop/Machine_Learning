import random

# ------------------ Node ------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# ------------------ Gini ------------------
def gini(y):
    impurity = 1
    for c in set(y):
        p = y.count(c) / len(y)
        impurity -= p ** 2
    return impurity


# ------------------ Split ------------------
def split(X, y, feature, threshold):
    left_X, left_y, right_X, right_y = [], [], [], []

    for i in range(len(X)):
        if X[i][feature] < threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])

    return left_X, left_y, right_X, right_y


# ------------------ Best Split ------------------
def best_split(X, y, max_features):
    n_features = len(X[0])
    features = random.sample(range(n_features), max_features)

    best_gini = float('inf')
    best = None

    for feature in features:
        values = set(row[feature] for row in X)

        for val in values:
            left_X, left_y, right_X, right_y = split(X, y, feature, val)

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            g = (len(left_y)/len(y)) * gini(left_y) + \
                (len(right_y)/len(y)) * gini(right_y)

            if g < best_gini:
                best_gini = g
                best = (feature, val)

    return best


# ------------------ Build Tree ------------------
def build_tree(X, y, depth=0, max_depth=3, max_features=1):
    if len(set(y)) == 1 or depth >= max_depth:
        return Node(value=max(set(y), key=y.count))

    split_point = best_split(X, y, max_features)

    if split_point is None:
        return Node(value=max(set(y), key=y.count))

    feature, threshold = split_point
    left_X, left_y, right_X, right_y = split(X, y, feature, threshold)

    left = build_tree(left_X, left_y, depth+1, max_depth, max_features)
    right = build_tree(right_X, right_y, depth+1, max_depth, max_features)

    return Node(feature, threshold, left, right)


# ------------------ Predict Tree ------------------
def predict_tree(node, x):
    if node.value is not None:
        return node.value

    if x[node.feature] < node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)


# ------------------ Bootstrap ------------------
def bootstrap(X, y):
    n = len(X)
    X_sample, y_sample = [], []

    for _ in range(n):
        i = random.randint(0, n-1)
        X_sample.append(X[i])
        y_sample.append(y[i])

    return X_sample, y_sample


# ------------------ Random Forest ------------------
class RandomForest:
    def __init__(self, n_trees=5, max_depth=3, max_features=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap(X, y)
            tree = build_tree(X_sample, y_sample,
                              max_depth=self.max_depth,
                              max_features=self.max_features)
            self.trees.append(tree)

    def predict(self, X):
        results = []
        for x in X:
            preds = [predict_tree(tree, x) for tree in self.trees]
            final = max(set(preds), key=preds.count)  # majority vote
            results.append(final)
        return results


# ------------------ Example ------------------
X = [
    [22, 20],
    [25, 25],
    [35, 50],
    [45, 80]
]

y = ["No", "No", "Yes", "Yes"]

rf = RandomForest(n_trees=5, max_depth=3, max_features=1)
rf.fit(X, y)

print(rf.predict([[30, 40]])) 