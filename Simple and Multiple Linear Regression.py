#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages and dataset
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

data = pd.read_csv('Advertising.csv', index_col = 0)
#Since, in the data the 0th index ('Unnamed_Column') represents index. Specifying that in the dataframe so the he column is treated as an index instead of a seperate column


# In[2]:


#Checking if the data is imported properly
data.head()


# In[3]:


#Checking dimensions of the data
data.shape


# In[4]:


#Checking for Null Values
data.isnull().sum()


# In[5]:


#Plotting a catter-plot to understand the data better, presently doing a simple linear regression (1 predictor).
plt.scatter(data['TV'], data['Sales'])


# In[6]:


plt.scatter(data['Newspaper'], data['Sales'])


# In[7]:


plt.scatter(data['Radio'], data['Sales'])


# In[8]:


plt.boxplot(data['TV'])


# In[9]:


plt.boxplot(data['Radio'])


# In[10]:


plt.boxplot(data['Newspaper'])


# In[11]:


#Outliers present in data, since there're only 2 records removing them won't make much of a difference
drop_index = data[data['Newspaper'] > 100].index
data.drop(drop_index, inplace = True)
plt.boxplot(data['Newspaper'])


# In[12]:


#Since the Last column specifies the Y (dependent) variable, seperating out the independent and dependent variable.
X = data.iloc[:, :-1]
y = data['Sales']


# In[13]:


#Splitting data into training and testing
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, shuffle = True)


# In[14]:


#Having a look at the splitted data
X_Train.head()


# In[15]:


X_Test.head()


# In[16]:


y_Train.head()


# In[17]:


y_Test.head()


# In[18]:


#Building Simple Linear Model using sklearn
model = LinearRegression().fit(X_Train.iloc[:,0].values.reshape(-1, 1), y_Train)


# In[19]:


#Looking at the co-efficients
print("Simple Linear Regression Coefficients (for TV advertising):")
print(f'Sales = {model.intercept_} + {model.coef_[0]} (TV)')


# So according to the model, when there is no TV marketing done then the sales value must ideally be 7.2350 and for every $1000 spent in TV Marketing there is an increase of 0.046 in Sales
# 
# NOTE: Sales is in thousands

# In[20]:


#Predicting values
Pred = model.predict(X_Test.iloc[:, 0].values.reshape(-1, 1))


# In[21]:


#Plotting the regression Line
plt.figure(figsize = (16, 8))

plt.scatter(X['TV'], y)

plt.plot(X_Test.iloc[:, 0], Pred, alpha=2)


# In[22]:


#Displaying Metrics
data.iloc[:, 0].shape
est = sm.OLS(data['Sales'], sm.add_constant(data['TV']))
est = est.fit()
print(est.summary())


# In[23]:


#9 Some more inights
print ('MAE:', metrics.mean_absolute_error(Pred, y_Test))
print ('RMSE:', np.sqrt(metrics.mean_squared_error(Pred, y_Test)))
print ('R-Squared:', metrics.r2_score(Pred, y_Test))


# As seen above, the accuracy is ~60% - 70% which is quite low.
# 
# ### Why Multiple Linear Regression?
# Simple linear regression is a useful approach for predicting a response on the basis of a single predictor variable. However, in practice we often have more than one predictor. For example, in the Advertising data, we have examined the relationship between sales and TV advertising. We also have data for the amount of money spent advertising on the radio and in newspapers, and we may want to know whether either of these two media is associated with sales.
# 
# One option is to run three separate simple linear regressions, each of which uses a different advertising medium as a predictor. For instance, we can fit a simple linear regression to predict sales on the basis of the amount spent on radio advertisements.
# 
# However, the approach of fitting a separate simple linear regression model for each predictor is not entirely satisfactory. First of all, it is unclear how to make a single prediction of sales given levels of the three advertising media budgets, since each of the budgets is associated with a separate regression equation. Second, each of the three regression equations ignores the other two media in forming estimates for the regression coefficients.
# Instead of fitting a separate simple linear regression model for each predictor, a better approach is to extend the simple linear regression model so that it can directly accommodate multiple predictors. We can do this by giving each predictor a separate slope coefficient in a single model.

# In[24]:


model_mlr = LinearRegression().fit(X_Train, y_Train)


# In[25]:


Pred2 = model_mlr.predict(X_Test)


# In[26]:


print("The linear model is: Sales = {:.5} + {:.5} (TV) + {:.5} (Radio) + {:.5} (Newspaper)".format(model_mlr.intercept_, model_mlr.coef_[0], model_mlr.coef_[1], model_mlr.coef_[2]))


# In[27]:


model_mlr.score(X_Train, y_Train)


# In[28]:


# Print out the statistics
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

