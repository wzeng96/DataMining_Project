#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('classic')
#https://datahub.io/cryptocurrency/bitcoin#data-cli
#%% [markdown]
import os
dirpath = os.getcwd() 
print("current directory is : " + dirpath)
filepath = os.path.join( dirpath, 'bitcoin_csv.csv')
print(filepath)

import xlrd
 
bitcoin = pd.read_csv(filepath)
fp = os.path.join( dirpath, 'bitcoin_csv.csv')
btc = pd.read_csv(fp)

#%%
#Data cleaning
btc = bitcoin.dropna()
btc = btc.dropna()
#add price change (percent) columns
l = [True]
percl = [0]
p_arr = btc.iloc[:,5].values
for i in range(1,len(btc)):
    l.append(p_arr[i] > p_arr[i-1])
    percl.append((p_arr[i]-p_arr[i-1])/p_arr[i-1])
btc['pc'] = l
btc['pcp'] = percl

btc= btc[btc['exchangeVolume(USD)'] != 0]

btc.columns
#%%
plt.plot(btc['txVolume(USD)'], label='totalVol')
#plt.xlabel('txV')
#plt.ylabel('amount')
#filepath = os.path.join( dirpath,'hist_age.png')
#plt.savefig(filepath)
plt.show()


#%% Correlation
import seaborn as sns
plt.figure(figsize=(12,10))
cor = btc.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
#filepath = os.path.join( dirpath ,'corr_b.png')
#plt.savefig(filepath)
plt.show()


plt.figure(figsize=(12,9))
plt.plot(btc['generatedCoins'].values)

# %%
#Get X and y variable. Y is one day late
XX = btc[btc.columns.difference(['date', 'marketcap(USD)','price(USD)','txVolume(USD)','adjustedTxVolume(USD)','exchangeVolume(USD)','medianTxValue(USD)'])].iloc[1:,:]
X = XX.values
y = btc.iloc[:-1, 5:6].values
yp = btc.iloc[:-1, -2:-1].values
# XX.columns()
# X.columns
# y.columns
# yp.columns

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, yp, test_size = 0.2, random_state = 0)

# %%

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
btc.columns=btc.columns.str.strip().str.lower().str.replace('(','').str.replace('U','').str.replace('S','').str.replace('D','').str.replace(')','')

btc.head()
#%%
# Linear Regression
from statsmodels.formula.api import ols

modelPrice = ols(formula = 'priceusd ~ txvolumeusd + adjustedtxvolumeusd + txcount + generatedcoins + fees + activeaddresses + averagedifficulty + mediantxvalueusd + blocksize', data=btc).fit()
print( modelPrice.summary() )

#%%
modelpredicitons = pd.DataFrame( columns=['price_ALLlm'], data= modelPrice.predict(btc)) 
print(modelpredicitons.shape)
print( modelpredicitons.head() )

# %%
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X = btc[['txvolumeusd','adjustedtxvolumeusd', 'txcount', 'generatedcoins', 'fees', 'activeaddresses', 'averagedifficulty','mediantxvalueusd','blocksize']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ] # list comprehension

# View results using print
print(vif)

#%%
from sklearn import linear_model

full_split = linear_model.LinearRegression() # new instancew
full_split.fit(X_train, y_train)
y_pred = full_split.predict(X_test)
full_split.score(X_test, y_test)

print('score:', full_split.score(X_test, y_test)) # 0.8585809341981796
print('intercept:', full_split.intercept_) # -1835.62848196
print('coef_:', full_split.coef_)  # [ 2.52354996e-02  8.16512793e-10  6.10099655e+00 -5.02771703e-05
   # 3.80343776e+00  1.70709262e-01  3.33882792e-02  2.54640191e-03
   # -1.68412021e+02 -9.78869937e+02 -3.33673149e-02]


#%% 
from sklearn.model_selection import cross_val_score
xbtc = btc[['txvolumeusd','adjustedtxvolumeusd', 'txcount', 'activeaddresses','blocksize']]
# print(xbtc.head())
ybtc = btc['priceusd']
full_cv = linear_model.LinearRegression()
cv_results = cross_val_score(full_cv, xbtc, ybtc, cv=10)
print(cv_results) # [0.99982467 0.99014869 0.98341804 0.99957296 0.99898658]
np.mean(cv_results) # 0.9943901862799376
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))
















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


#%%
# 王芷霖previous code
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
plt.plot(y_pred,label = 'pred')
plt.plot(y_test,label = 'actual')
plt.legend()
plt.show()

reg.score(X_test,y_test)

# %%
from sklearn.ensemble import RandomForestRegressor
RFreg = RandomForestRegressor(n_estimators = 100, random_state = 0)
RFreg.fit(X_train, y_train)
plt.plot(RFreg.predict(X_test),label = 'pred')
plt.plot(y_test,label = 'actual')
plt.legend()
plt.show()

RFreg.score(X_test,y_test)

# %%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xp_train, yp_train)
# Predicting the Test set results
yp_pred = classifier.predict(Xp_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yp_test, yp_pred)

cm

# %%
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(Xp_train, yp_train)

# Predicting the Test set results
yp_pred = classifier.predict(Xp_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yp_test, yp_pred)
cm

# %%


# %%
