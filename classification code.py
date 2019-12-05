#%%
# Logistic Regression
import statsmodels.api as sm  
from statsmodels.formula.api import glm

modelPriceLogit = glm(formula='pc ~ adjustedtxvolumeusd + txcount + generatedcoins + fees + activeaddresses + averagedifficulty + mediantxvalueusd', data=btc, family=sm.families.Binomial()).fit()
print( modelPriceLogit.summary() )

#%% 
# Logistic Regression cut off
cut_off = 0.3
modelprediciton = pd.DataFrame( columns=['pc_AllLog'], data= modelPriceLogit.predict(btc)) 
modelprediciton['pc'] = modelPriceLogit.predict(btc)

modelprediciton['pcLogitAll'] = np.where(modelprediciton['pc_AllLog'] > cut_off, 1, 0)

print(pd.crosstab(btc.pc, modelprediciton.pcLogitAll,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

#%%
# KNN
xpc = btc[['adjustedtxvolumeusd', 'txcount', 'generatedcoins', 'fees', 'activeaddresses', 'averagedifficulty', 'mediantxvalueusd']]
ypc = btc['pc']
# print(type(xpc))
# print(type(ypc))


# %%
# k values
mrroger = 3
mrroger2 = 6
mrroger3 = 9

#%%
# unscaled When k = 3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn.fit(xpc,ypc)
y_pred = knn.predict(xpc)
print(y_pred)
knn.score(xpc,ypc)

import numpy as np
knn_cv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv, xpc, ypc, cv=5)
print(cv_results) 
np.mean(cv_results) 

# %%
# unscaled When k = 6
knn_cv = KNeighborsClassifier(n_neighbors=mrroger2) # instantiate with n value given
cv_results = cross_val_score(knn_cv, xpc, ypc, cv=5)
print(cv_results) 
np.mean(cv_results) 

#%%
# scale When k = 3
from sklearn.preprocessing import scale
xspc = pd.DataFrame( scale(xpc), columns=xpc.columns ) 
yspc = ypc.copy() 

knn_scv = KNeighborsClassifier(n_neighbors=mrroger) 

scv_results = cross_val_score(knn_scv, xspc, yspc, cv=5)
print(scv_results) 
np.mean(scv_results) 

# %%
# scaled When k = 6
knn_scv = KNeighborsClassifier(n_neighbors=mrroger2) 

scv_results = cross_val_score(knn_scv, xspc, yspc, cv=5)
print(scv_results) 
np.mean(scv_results) 

#%%
# Compare KNN with logistic regression accuracy
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xspc, yspc)
lr.score(xspc, yspc) # accuracy score

# Compare with KNN, logistic regression with scaled data has 55.69% accuracy, while KNN with scaled data when k = 3 is 47.27% accuracy.

#%%
# Classification Decision Tree
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

# Instantiate dtree
dtree_pc1 = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtree_pc1.fit(Xp_train,yp_train)
# Predict test set labels
yp_pred = dtree_pc1.predict(Xp_test)
# Evaluate test-set accuracy
print(accuracy_score(yp_test, yp_pred))
print(confusion_matrix(yp_test, yp_pred))
print(classification_report(yp_test, yp_pred))

#%%
logitreg_pc1 = LogisticRegression(random_state=1)
logitreg_pc1.fit(Xp_train, yp_train)

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

plot_decision_regions(Xp_test.values, yp_test.values, clf=logitreg_pc1, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2: 3} )

plt.xlabel(X_test.columns[0])
plt.ylabel(X_test.columns[1])
plt.title(logitreg_pc1.__class__.__name__)
plt.show()

plot_decision_regions(X_test.values, y_test.values, clf=dtree_pc1, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2: 3} )
plt.xlabel(X_test.columns[0])
plt.ylabel(X_test.columns[1])
plt.title(dtree_pc1.__class__.__name__)
plt.show()
