import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from pygam import LogisticGAM, s, f


mdls = {
    'LogReg': LogisticRegression(penalty=None, max_iter=1000),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'GNB': GaussianNB()
}

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

for nm, mdl in mdls.items():
    mdl.fit(X_train, y_train.values.ravel())
    y_pred = mdl.predict(X_test)
    y_prob = mdl.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    print(f'{nm} - Acc: {acc:.4f}, LogLoss: {loss:.4f}')






gam = LogisticGAM(f(0) + f(1) + s(2) + s(3))
gam.fit(X_train, y_train)

y_p_gam = gam.predict(X_test)
y_prob_gam = gam.predict_proba(X_test)

acc_gam = accuracy_score(y_test, y_p_gam)
loss_gam = log_loss(y_test, y_prob_gam)

print(f'GAM - Acc: {acc_gam:.4f}, LogLoss: {loss_gam:.4f}')







features = ['Class', 'Sex', 'Age', 'Fare']
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdep, conf = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], conf, c='r', ls='--')
    ax.set_title(features[i])

plt.tight_layout()
plt.show()