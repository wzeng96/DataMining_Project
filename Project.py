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

#%% Correlation
import seaborn as sns
plt.figure(figsize=(12,10))
cor = btc.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
#filepath = os.path.join( dirpath ,'corr_b.png')
#plt.savefig(filepath)
plt.show()


plt.figure(figsize=(12,9))
# plt.plot(btc['generatedCoins'].values)

# %%
#Get X and y variable. Y is one day late
btc.columns=btc.columns.str.strip().str.lower().str.replace('(','').str.replace('U','').str.replace('S','').str.replace('D','').str.replace(')','')

XX = btc[btc.columns.difference(['date', 'marketcapusd','priceusd','txVolumeusd','adjustedTxVolumeusd','exchangeVolumeusd','medianTxValueusd'])].iloc[1:,:]
X = XX.values
y = btc.iloc[:-1, 5:6].values
# print(y)
yp = btc.iloc[:-1, -2:-1].values
# XX.columns()
# X.columns
# y.columns
# yp.columns

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, yp, test_size = 0.2, random_state = 123)

# %%

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.fit_transform(y_test)

btc.head()
# xbtc = btc[['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'activeaddresses']]
# # print(xbtc.head())
# ybtc = btc['priceusd']
# xbtcs = pd.DataFrame( scale(xbtc), columns=xbtc.columns )
# priceusd = pd.DataFrame(btc['priceusd'])
# priceusd

#%%
from sklearn import linear_model

full_split = linear_model.LinearRegression() # new instancew
full_split.fit(X_train, y_train)
y_pred = full_split.predict(X_test)
full_split.score(X_test, y_test)

#print(y_pred[0:5])

print('score:', full_split.score(X_test, y_test)) # 0.9425689619048208
print('intercept:', full_split.intercept_) # [2535.11953212]
print('coef_:', full_split.coef_)  # [ 1399.60277109  1081.09477569   840.17730788   172.53603703
                                   #  -196.81423833  1122.51188693  -456.76237218  -333.06092781
                                   #  150.98440138   318.11769235    37.67772706     2.21087506
                                   #  -72.91491501 -1075.72250299   244.68387439]

#%%
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
# %%
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
Xvif = btc[['txvolumeusd','adjustedtxvolumeusd', 'txcount', 'generatedcoins', 'fees', 'activeaddresses', 'averagedifficulty','mediantxvalueusd', 'blocksize']]
Xvif['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = Xvif.columns
vif["VIF"] = [ variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1]) ] # list comprehension

# View results using print
print(vif)

# Therefore, txvolume, adjustedtxvolumn, txcount and activeaddresses and blocksize are highly correlated with price

#%%
## Adjusted Linear Model
xbtc = btc[['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'activeaddresses', 'blocksize']]
ybtc = btc['priceusd']

Xad_train, Xad_test, yad_train, yad_test = train_test_split(xbtc, ybtc, test_size = 0.2, random_state = 1)

scad_X = StandardScaler()
Xad_train = scad_X.fit_transform(Xad_train)
Xad_test = scad_X.transform(Xad_test)


ad_split = linear_model.LinearRegression() # new instancew
ad_split.fit(Xad_train, yad_train)
yad_pred = ad_split.predict(Xad_test)
ad_split.score(Xad_test, yad_test)

print('score:', ad_split.score(Xad_test, yad_test)) # 0.8906185214311232
print('intercept:', ad_split.intercept_) # 204.34242221557088
print('coef_:', ad_split.coef_)  # [-2.00364500e-07  1.67363753e-06 -2.28343962e-02  1.05130901e-02 -4.24595526e-07]
print(yad_pred[0:5])
plt.scatter(yad_test, yad_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

#%% 
# Super Confused!!!!!!!!!!!!!!!!!!!!!Cross-Validation
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict

full_cv = linear_model.LinearRegression()
cv_results = cross_val_score(full_cv, Xad_train, yad_train, cv=10)
print(cv_results) 
np.mean(cv_results) 
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

ycv_pred = cross_val_predict(full_cv, Xad_train, yad_train, cv=10)
plt.scatter(yad_train, ycv_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

from sklearn import metrics
accuracy = metrics.r2_score(yad_train, ycv_pred)
print( 'Cross-Predicted Accuracy:', accuracy)




#%%
# Regression Tree
# import seaborn as sns
# sns.set()
# sns.pairplot(xbtc)

#%%
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import mean_squared_error as MSE  # Import mean_squared_error as MSE
# Split data into 80% train and 20% test
#X_train, X_test, y_train, y_test= train_test_split(xbtcs, ybtcs, test_size=0.2,random_state=1)
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree0 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1,random_state=1) # set minimum leaf to contain at least 10% of data points
# DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
#     max_leaf_nodes=None, min_impurity_decrease=0.0,
#     min_impurity_split=None, min_samples_leaf=0.13,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=3, splitter='best')


regtree0.fit(Xad_train, yad_train)  # Fit regtree0 to the training set
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# evaluation
yad_pred = regtree0.predict(Xad_test)  # Compute y_pred
mse_regtree0 = MSE(yad_test, yad_pred)  # Compute mse_regtree0
rmse_regtree0 = mse_regtree0 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0)) # 1860.22

#%%
# Let us compare the performance with OLS
from sklearn import linear_model
olsbtc = linear_model.LinearRegression() 
olsbtc.fit( Xad_train, yad_train )

y_pred_ols = olsbtc.predict(Xad_test)  # Predict test set labels/values

mse_ols = MSE(yad_test, y_pred_ols)  # Compute mse_ols
rmse_ols = mse_ols**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

# %%

# Compare the tree with CV
regtree1 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1, random_state=1)

# Evaluate the list of MSE ontained by 10-fold CV

from sklearn.model_selection import cross_val_score
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(regtree1, Xad_train, yad_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree1.fit(Xad_train, yad_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree1.predict(Xad_train)  # Predict the labels of training set
y_predict_test = regtree1.predict(Xad_test)  # Predict the labels of test set

print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE:', MSE(yad_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(yad_test, y_predict_test)**(0.5) )   # Test set MSE 

regtree1.score(Xad_test, yad_test)
#%% Try prediction
forecast_out = 15
#Create another column (the target ) shifted 'n' units up
df = btc[['priceusd']]
df['Prediction'] = df[['priceusd']].shift(-forecast_out)
#print the new data set
print(df.tail())

X = np.array(df.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)

y = np.array(df['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

lr = LinearRegression()
lr_predict = lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)
#%% [markdown]
#
# #  Bias-variance tradeoff  
# high bias: underfitting  
# high variance: overfitting, too much complexity  
# Generalization Error = (bias)^2 + Variance + irreducible error  
# 
# Solution: Use CV  
# 
# 1. If CV error (average of 10- or 5-fold) > training set error  
#   - high variance
#   - overfitting the training set
#   - try to decrease model complexity
#   - decrease max depth
#   - increase min samples per leaf
#   - get more data
# 2. If CV error approximates the training set error, and greater than desired error
#   - high bias
#   - underfitting the training set
#   - increase max depth
#   - decrease min samples per leaf
#   - use or gather more relevant features

#%%
# Graphing the tree
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
# import os
# dirpath = os.getcwd() # print("current directory is : " + dirpath)
path2add = 'd:/Download/George Washington University/Fall 2019/6103/DataMining_Project'
filepath = os.path.join( dirpath, path2add ,'tree1')
export_graphviz(regtree1, out_file = filepath+'.dot' , feature_names =['txvolumeusd', 'adjustedtxvolumeusd', 'txcount', 'activeaddresses', 'blocksize']) 

# import pydot

# (graph,) = pydot.graph_from_dot_file(filepath)
# graph.write_png(filepath + '.png')


#%%
# Linear Regression
## Full Model
from statsmodels.formula.api import ols

modelPrice = ols(formula = 'priceusd ~ txvolumeusd + adjustedtxvolumeusd + txcount + generatedcoins + fees + activeaddresses + averagedifficulty + mediantxvalueusd + blocksize', data=btc).fit()
print( modelPrice.summary() )

#%%
## Adjusted Model
modelPricead = ols(formula = 'priceusd ~ txvolumeusd + adjustedtxvolumeusd + txcount + generatedcoins + fees + activeaddresses + averagedifficulty + mediantxvalueusd', data=btc).fit()
print( modelPricead.summary() )

# We have a R^2 of 95.9%, which is good, however, this linear model might overfit the data.

#%%
modelpredicitons = pd.DataFrame( columns=['price_ALLlm'], data= modelPricead.predict(btc)) 
print(modelpredicitons.shape)
print( modelpredicitons.head() )

# %%
# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X = btc[['txvolumeusd','adjustedtxvolumeusd', 'txcount', 'generatedcoins', 'fees', 'activeaddresses', 'averagedifficulty','mediantxvalueusd']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ] # list comprehension

# View results using print
print(vif)

# Therefore, txvolume, adjustedtxvolumn, txcount and activeaddresses are highly correlated with price

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
from sklearn.preprocessing.data import scale
cm = confusion_matrix(yp_test, yp_pred)
cm

# %%


# %%
