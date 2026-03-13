import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from statsmodels.api import Logit, add_constant
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

np.random.seed(4771)

N = 80

X1 = np.random.uniform(-2, 2, N)
X2 = np.random.uniform(-2, 2, N)

eta = 4*(X1**2) - 2*X2 + 1.5*X1*X2
p = 1 / (1 + np.exp(-eta))
y = (p > 0.5).astype(int)

X = np.column_stack((X1, X2))

X_c = add_constant(X)
mdl_1 = Logit(y, X_c).fit()
print(mdl_1.summary())

x1_g, x2_g = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid = np.column_stack((x1_g.ravel(), x2_g.ravel()))
grid_c = add_constant(grid)
p_1 = mdl_1.predict(grid_c).reshape(x1_g.shape)

plt.figure()
plt.contourf(x1_g, x2_g, p_1, alpha=0.4, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.show()





poly = PolynomialFeatures(degree=2, include_bias=True)
X_p = poly.fit_transform(X)
mdl_2 = Logit(y, X_p).fit()
print(mdl_2.summary())





poly_nb = PolynomialFeatures(degree=2, include_bias=False)
X_p_nb = poly_nb.fit_transform(X)
grid_p = poly_nb.transform(grid)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, c_val in zip(axs, [0.01, 1, 100]):
    lr = LogisticRegression(C=c_val, penalty='l2')
    lr.fit(X_p_nb, y)
    p_3 = lr.predict_proba(grid_p)[:, 1].reshape(x1_g.shape)
    ax.contourf(x1_g, x2_g, p_3, alpha=0.4, cmap='bwr')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    ax.set_title(f'C={c_val}')
plt.show()






import matplotlib as mpl
from sklearn.inspection import DecisionBoundaryDisplay

def plot_ell(mu, cov, col, ax):
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    ang = 180 * np.arctan(u[1] / u[0]) / np.pi
    ell = mpl.patches.Ellipse(mu, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, angle=180 + ang, facecolor=col, edgecolor="k", lw=2)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)

def plot_res(est, X_data, y_data, ax):
    DecisionBoundaryDisplay.from_estimator(est, X_data, response_method="predict_proba", plot_method="pcolormesh", ax=ax, cmap="bwr", alpha=0.3)
    DecisionBoundaryDisplay.from_estimator(est, X_data, response_method="predict_proba", plot_method="contour", ax=ax, alpha=1.0, levels=[0.5])
    
    y_p = est.predict(X_data)
    X_r, y_r = X_data[y_data == y_p], y_data[y_data == y_p]
    X_w, y_w = X_data[y_data != y_p], y_data[y_data != y_p]
    
    ax.scatter(X_r[:, 0], X_r[:, 1], c=y_r, s=20, cmap="bwr", alpha=0.5, edgecolors='k')
    ax.scatter(X_w[:, 0], X_w[:, 1], c=y_w, s=30, cmap="bwr", alpha=0.9, marker="x")
    ax.scatter(est.means_[:, 0], est.means_[:, 1], c="yellow", s=200, marker="*", edgecolor="k")

    covs = [est.covariance_] * 2 if isinstance(est, LinearDiscriminantAnalysis) else est.covariance_
    plot_ell(est.means_[0], covs[0], "blue", ax)
    plot_ell(est.means_[1], covs[1], "red", ax)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

lda = LinearDiscriminantAnalysis(store_covariance=True)
lda.fit(X, y)
plot_res(lda, X, y, axs[0])
axs[0].set_title("LDA")

qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(X, y)
plot_res(qda, X, y, axs[1])
axs[1].set_title("QDA")

plt.show()