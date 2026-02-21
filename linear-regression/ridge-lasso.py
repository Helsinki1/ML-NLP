import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path
from sklearn.linear_model import ridge_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV


# Step 1
df = pd.read_csv('Hitters_train.csv')
scalar = StandardScaler()
scaled_data = scalar.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)


# Step 2
y_train = df_scaled["16"]
x_train = df_scaled.drop(columns=["16"])
X_train_with_constant = sm.add_constant(x_train)

model = sm.OLS(y_train, X_train_with_constant)
results = model.fit()
print(results.summary())

''' 
OLS Regression Results                            
==============================================================================
Dep. Variable:                     16   R-squared:                       0.549
Model:                            OLS   Adj. R-squared:                  0.511
Method:                 Least Squares   F-statistic:                     14.46
Date:                Fri, 20 Feb 2026   Prob (F-statistic):           4.01e-25
Time:                        22:39:46   Log-Likelihood:                -211.27
No. Observations:                 207   AIC:                             456.5
Df Residuals:                     190   BIC:                             513.2
Df Model:                          16                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       1.041e-17      0.049   2.14e-16      1.000      -0.096       0.096
0             -0.5246      0.230     -2.286      0.023      -0.977      -0.072
1              0.4424      0.275      1.610      0.109      -0.100       0.984
2              0.0154      0.143      0.108      0.914      -0.266       0.297
3             -0.0134      0.185     -0.072      0.942      -0.379       0.352
4              0.1753      0.172      1.022      0.308      -0.163       0.514
5              0.2318      0.101      2.289      0.023       0.032       0.432
6             -0.0154      0.153     -0.100      0.920      -0.318       0.287
7             -0.9119      0.811     -1.125      0.262      -2.511       0.687
8              1.2905      1.194      1.081      0.281      -1.065       3.645
9              0.0225      0.342      0.066      0.948      -0.653       0.698
10             0.7458      0.614      1.214      0.226      -0.466       1.957
11            -0.3770      0.609     -0.619      0.537      -1.578       0.824
12            -0.3141      0.233     -1.345      0.180      -0.775       0.146
13             0.2132      0.054      3.951      0.000       0.107       0.320
14             0.0639      0.086      0.744      0.458      -0.106       0.233
15            -0.0378      0.076     -0.500      0.618      -0.187       0.111
==============================================================================
Omnibus:                       26.596   Durbin-Watson:                   2.062
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               46.495
Skew:                           0.682   Prob(JB):                     8.01e-11
Kurtosis:                       4.879   Cond. No.                         84.5
==============================================================================
'''

# Comments: Out of the 16 features, there are 3 significant predictors (0, 5, 13)
# Predictor #0: p=0.023 (<0.05), |t|=2.289 (>2), std err = 0.230 (quite small)
# Predictor #5: p=0.023 (<0.05), |t|=2.289 (>2), std err = 0.101 (quite small)
# Predictor #13: p=0.000 (<0.05), |t|=3.951 (>2), std err = 0.054 (very small)
# Since R^2=0.549, the model explains 54.9% of the variance and the model fit is not great.



# Step 3
alphas_lasso = np.logspace(-2, 1, 100)
alphas, coeffs, _ = lasso_path(x_train, y_train, alphas=alphas_lasso)

plt.figure(figsize=(10, 6))
for i in range(coeffs.shape[0]):
    plt.plot(alphas, coeffs[i], label=f'Feature {i}')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Lasso Path')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()



# Step 4
alphas_ridge = np.logspace(-2, 2, 100)
coeffs = []

for alpha in alphas_ridge:
    coeff = ridge_regression(x_train, y_train, alpha=alpha)
    coeffs.append(coeff)

coeffs = np.array(coeffs)

plt.figure(figsize=(10, 6))
for i in range(coeffs.shape[1]):
    plt.plot(alphas_ridge, coeffs[:, i], label=f'Feature {i}')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficients')
plt.title('Ridge Path')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()



# Step 5
df = pd.read_csv('Hitters_test.csv')
scaled_data = scalar.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
y_test = df_scaled["16"]
x_test = df_scaled.drop(columns=["16"])
X_test_with_constant = sm.add_constant(x_test)

ols = LinearRegression()
ols.fit(x_train, y_train)
ols_score = ols.score(x_test, y_test)

ridge_cv = RidgeCV(alphas=alphas_ridge, cv=5)
ridge_cv.fit(X_train_with_constant, y_train)
ridge_cv_score = ridge_cv.score(X_test_with_constant, y_test)
ridge_penalty = ridge_cv.alpha_

lasso_cv = LassoCV(alphas=alphas_lasso, cv=5)
lasso_cv.fit(X_train_with_constant, y_train)
lasso_cv_score = lasso_cv.score(X_test_with_constant, y_test)
lasso_penalty = lasso_cv.alpha_

print(f"OLS R^2 Score: {ols_score:.4f}")
print(f"Ridge R^2 Score: {ridge_cv_score:.4f}")
print(f"Lasso R^2 Score: {lasso_cv_score:.4f}")
print(f"Ridge Penalty: {ridge_penalty:.4f}")
print(f"Lasso Penalty: {lasso_penalty:.4f}")

'''
OLS R^2 Score: 0.4410
Ridge R^2 Score: 0.4396
Lasso R^2 Score: 0.4359
Ridge Penalty: 14.1747
Lasso Penalty: 0.0433
'''