import math

# -----------------------------
# DATASET
# -----------------------------

x = [2, 4, 6, 8, 10]

y = [0, 0, 1, 1, 1]

# -----------------------------
# INITIAL WEIGHTS
# -----------------------------

w = 0.1
b = 0.1

learning_rate = 0.1
epochs =50

# -----------------------------
# SIGMOID FUNCTION
# -----------------------------

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# -----------------------------
# TRAINING
# -----------------------------

for epoch in range(epochs):

    total_loss = 0

    print("\n========================")
    print("EPOCH:", epoch + 1)
    print("========================")

    for i in range(len(x)):

        # -----------------------------
        # FORWARD PROPAGATION
        # -----------------------------

        z = (w * x[i]) + b

        pred = sigmoid(z)

        # -----------------------------
        # LOSS
        # -----------------------------

        loss = (y[i] - pred) ** 2

        total_loss += loss

        # -----------------------------
        # BACKPROPAGATION
        # -----------------------------

        # dL/dPred
        dL_dpred = 2 * (pred - y[i])

        # sigmoid derivative
        dpred_dz = pred * (1 - pred)

        # dL/dz
        dL_dz = dL_dpred * dpred_dz

        # dL/dw
        dL_dw = dL_dz * x[i]

        # dL/db
        dL_db = dL_dz

        # -----------------------------
        # GRADIENT DESCENT
        # -----------------------------

        w = w - (learning_rate * dL_dw)

        b = b - (learning_rate * dL_db)

        # -----------------------------
        # PRINT EACH INPUT RESULT
        # -----------------------------

        print("\nInput:", x[i])
        print("Actual Output:", y[i])

        print("Prediction:", round(pred, 4))

        print("Loss:", round(loss, 4))

        print("Gradient dw:", round(dL_dw, 4))
        print("Gradient db:", round(dL_db, 4))

        print("Updated Weight:", round(w, 4))
        print("Updated Bias:", round(b, 4))

    print("\nTotal Loss:", round(total_loss, 4))

# -----------------------------
# FINAL TRAINED VALUES
# -----------------------------

print("\n========================")
print("FINAL TRAINED PARAMETERS")
print("========================")

print("Final Weight:", round(w, 4))
print("Final Bias:", round(b, 4))

# -----------------------------
# PREDICT NEW DATA
# -----------------------------

print("\n========================")
print("NEW DATA PREDICTION")
print("========================")

new_x = 7

z = (w * new_x) + b

prediction = sigmoid(z)

print("Input:", new_x)
print("Prediction Probability:", round(prediction, 4))

if prediction >= 0.5:
    print("Predicted Class: 1")
else:
    print("Predicted Class: 0")