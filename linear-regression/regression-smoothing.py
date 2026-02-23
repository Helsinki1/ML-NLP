import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
     
np.random.seed(67)

N = 100
M = 50

x = np.linspace(0, 1, N)
Y = np.tile(np.sin(12 * (x + 0.2)) / (x + 0.2), [M,1]) + np.random.normal(0, 1, size=(M, N))
     

# Step 1

# cub spline regression
knots = np.array([i for i in np.arange(0, 1, 0.1)])

def data_matrix(x):
    x = np.asarray(x).ravel()
    N = len(x)
    K = len(knots)

    X = np.zeros((N, 4 + K))

    X[:, 0] = 1
    X[:, 1] = x
    X[:, 2] = x**2
    X[:, 3] = x**3

    for j, knot in enumerate(knots):
        diff = np.maximum(x - knot, 0) ** 3
        X[:, 4 + j] = diff

    return X

matrix = data_matrix(x)
residuals = np.zeros((M, N))

model = LinearRegression(fit_intercept=False)

for m in range(M):
    model.fit(matrix, Y[m])
    residuals[m] = model.predict(matrix)

cubic = np.var(residuals, axis=0, ddof=1)
     

# natural cubic spline regression
def diff(x, knot):
    return np.maximum(x - knot, 0) ** 3

def natural_matrix(x):
    x = np.asarray(x).ravel()
    N = len(x)
    K = len(knots)

    t_km1 = knots[-2]
    t_k   = knots[-1]
    denom = t_k - t_km1

    X = np.zeros((N, 2 + (K - 2)))

    X[:, 0] = 1
    X[:, 1] = x

    for j in range(K - 2):
        t_j = knots[j]

        X[:, 2 + j] = (
            diff(x, t_j)
            - diff(x, t_km1) * (t_k - t_j) / denom
            + diff(x, t_k)   * (t_km1 - t_j) / denom
        )
    return X

nat_matrix = natural_matrix(x)
res_nat = np.zeros((M,N))

for m in range(M):
    model.fit(nat_matrix, Y[m])
    res_nat[m] = model.predict(nat_matrix)

natural = np.var(res_nat, axis=0, ddof=1)










# Step 2

true_data = np.sin(12 * (x + 0.2)) / (x + 0.2)
cubic_mean = np.mean(residuals, axis = 0)
nat_mean = np.mean(res_nat, axis = 0)

cubic_bias = cubic_mean - true_data
nat_bias = nat_mean - true_data

# bias
plt.figure()
plt.plot(x, cubic_bias, label="Cubic Spline")
plt.plot(x, nat_bias, label="Natural Cubic Spline")
plt.title("Pointwise Bias")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

# variance
plt.figure()
plt.plot(x, cubic, label="Cubic Spline")
plt.plot(x, natural, label="Natural Cubic Spline")
plt.title("Pointwise Variance")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

# Observations: For Interior points, both cubic and natural splines behave very similarly.
# However for boundary points, regular cubic splines have higher variance but natural 
# splines have more bias. This is because natural spline forces linear behavior past 
# the boundary knots. This constraint stabilized edge behavior.



# Step 3


def tricube(u):
    mask = np.abs(u) < 1
    w = np.zeros_like(u)
    w[mask] = (1 - np.abs(u[mask])**3)**3
    return w

def local_lin(x, y, x_0, lam):
    u = (x - x_0) / lam
    X = np.vstack([np.ones_like(x), x - x_0]).T

    w = tricube(u)
    W = np.diag(w)

    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return beta[0]

def local_quad(x, y, x_0, lam):
    u = (x - x_0) / lam

    X = np.vstack([np.ones_like(x), x-x_0, (x-x_0)**2]).T

    w = tricube(u)
    W = np.diag(w)

    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return beta[0]

lam = 0.2

res_local_lin = np.zeros((M, N))
res_local_quad = np.zeros((M, N))

for m in range(M):
    for i, x_0 in enumerate(x):
        res_local_lin[m, i] = local_lin(x, Y[m], x_0, lam)
        res_local_quad[m, i] = local_quad(x, Y[m], x_0, lam)

        
mean_local_lin = np.mean(res_local_lin, axis=0)
mean_local_quad = np.mean(res_local_quad, axis=0)

bias_local_lin = mean_local_lin - true_data
bias_local_quad = mean_local_quad - true_data

var_ll = np.var(res_local_lin, axis=0, ddof=1)
var_lq = np.var(res_local_quad, axis=0, ddof=1)

plt.figure()
plt.plot(x, bias_local_lin, label="Local Linear")
plt.plot(x, bias_local_quad, label="Local Quadratic")
plt.title("Pointwise Bias")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

plt.figure()
plt.plot(x, var_ll, label="Local Linear")
plt.plot(x, var_lq, label="Local Quadratic")
plt.title("Pointwise Variance")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()


# Comments: At the interior, quadratic methods reduce bias because they better capture the 
# curvature. However, variance increases slighly. At exterior, local quadratic raises variance 
# because it tries to estimate curvature with limited data and local linear shows more bias at boundaries.