import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4])
Y = np.array([20, 38, 65, 78])

mean_x = np.mean(X)
mean_y = np.mean(Y)

slope = np.sum((X - mean_x) * (Y - mean_y)) / np.sum((X - mean_x) ** 2)
intercept = mean_y - slope * mean_x

y_line = slope * X + intercept
residual_error = Y - y_line
print(y_line)

r_squared = 1 - (np.sum((Y - y_line) ** 2) / np.sum((Y - mean_y) ** 2))

print(f'Slope: {slope:.2f}')
print(f'Intercept: {intercept:.2f}')
print(f'Residuals: {residual_error}')
print(f'R² score: {r_squared:.4f}')

# --- single clean plot ---
plt.scatter(X, Y, color='orange', label='Actual data')
plt.plot(X, y_line, color='blue', label='Best-fit line')

# residuals as vertical dotted lines
for i in range(len(X)):
    plt.vlines(X[i], y_line[i], Y[i], color='red', linestyle='dotted', label='Residual' if i == 0 else '')

plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()

plt.figure()
plt.scatter(X, residual_error, color='red')

plt.axhline(y=0, linestyle='--')

plt.xlabel("Hours")
plt.ylabel("Residuals")
plt.show()