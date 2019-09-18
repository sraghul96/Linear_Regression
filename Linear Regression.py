#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""# I. Preparing the dataset """
#1 Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#2 Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')


# In[2]:


#3 Classify dependent and independent variables
X = dataset.iloc[:,:-1].values  #independent variable YearsofExperience
y = dataset.iloc[:,-1].values  #dependent variable salary

print("\nIdependent Variable (Experience):\n", X)
print("\nDependent Variable (Salary):\n", y)

plt.scatter(X, y, alpha=1)


# In[3]:


#4 Creating training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 1/3,random_state = 0) 

print("\n\nTraining Set :\n----------------\n")
print("X = \n", X_train)
print("y = \n", y_train)

print("\n\nTest Set :\n----------------\n")
print("X = \n",X_test)
print("y = \n", y_test)


# In[4]:


#5 Train the Regressor with training set
model = sm.OLS(y_train, X_train).fit()

#6 predict the outcome of test sets
y_Pred = model.predict(X_test)


# In[5]:


#7 Mapping the Regression Line
plt.scatter(X, y, alpha=1)
plt.plot(X_test, y_Pred, c = 'blue')


# In[6]:


#8 Calculating the Accuracy of the predictions
from sklearn import metrics
print("Prediction Accuracy = ", metrics.r2_score(y_test, y_Pred))

# Print out the statistics
print(model.summary())


# In[7]:


#9 Some more inights
print ('MAE:', metrics.mean_absolute_error(y_Pred, y_test))
print ('RMSE:', np.sqrt(metrics.mean_squared_error(y_Pred, y_test)))
print ('R-Squared:', metrics.r2_score(y_Pred, y_test))


# In[8]:


#10 Comparing Actual and Predicted Salaries for he test set
print("\nActual vs Predicted Salaries \n-------------------------\n")
print("Actual :\n ", y_test)
print("Predicted :\n ", y_Pred)

