import numpy as np

X = np.array([1,2,3,4,5,6])
Y = np.array([0,0,1,1,1,1])

w = 0
b = 0
lr = 0.1

for i in range(1000):

    # Step 1: prediction
    pred = w * X + b

    # Step 2: sigmoid
    sigmoid = 1 / (1 + np.exp(-pred))

    # Step 3: clip
    sigmoid = np.clip(sigmoid, 1e-7, 1-1e-7)

    # Step 4: log loss
    log_loss = -(Y * np.log(sigmoid) + (1-Y) * np.log(1-sigmoid))

    # Step 5: gradient
    grad = (sigmoid - Y) * X

    # Step 6: update w and b
    w = w - lr * np.mean(grad)
    b = b - lr * np.mean(sigmoid - Y)

    pred = w * X + b

    # Step 7: convert to 0 or 1
    predicted = (sigmoid >= 0.5).astype(int)

    print(f"Loop {i+1} → predicted: {predicted} | actual: {Y} | loss: {np.mean(log_loss):.4f}")

    # Step 8: stop when prediction matches actual!
    if np.array_equal(predicted, Y):
        print(f"\n✅ Model learned at loop {i+1}!")
        break