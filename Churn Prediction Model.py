#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install imbalanced-learn

#pip install scikit-learn, in order to use sklearn


# Imbalanced-learn is a Python library specifically designed to address the challenges posed by imbalanced datasets in machine learning. 
# 
# Imbalanced datasets refer to datasets where the distribution of classes is highly skewed, with one class significantly outnumbering the other(s). This imbalance can lead to biased models that perform poorly on minority classes and affect the overall performance of a machine learning algorithm.
# 
# The imbalanced-learn library provides a variety of techniques and algorithms to tackle the issue of imbalanced datasets. Some key functionalities include:
# 
# Over-sampling: It provides methods to increase the number of samples in the minority class by generating synthetic samples, thus balancing the class distribution. One popular over-sampling method is Synthetic Minority Over-sampling Technique (SMOTE).
# 
# Under-sampling: It offers methods to reduce the number of samples in the majority class, thereby balancing the class distribution. Under-sampling methods randomly remove samples from the majority class.
# 
# Combination Sampling: It combines both over-sampling and under-sampling techniques to achieve better balance in the class distribution. SMOTEENN and SMOTETomek are examples of combination sampling methods, and other techniques.
# 
# By using imbalanced-learn, data scientists and machine learning practitioners can apply these techniques to preprocess their imbalanced datasets and improve the performance and accuracy of their models, especially when dealing with imbalanced class distributions.

# In[ ]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN


# **Understanding the libraries called**
# 
# 1.The SMOTEENN class is a combination sampling technique that combines the Synthetic Minority Over-sampling Technique (SMOTE) and Edited Nearest Neighbors (ENN) algorithm. It works in two steps:
# 
# Over-sampling using SMOTE: SMOTE generates synthetic samples for the minority class by interpolating between neighboring samples. It aims to increase the representation of the minority class and balance the class distribution.
# 
# Under-sampling using ENN: ENN removes noisy and borderline samples from the dataset by considering the nearest neighbors of each sample. It helps to remove instances that may be wrongly classified or contribute to overlapping between classes.
# 
# By combining these techniques, SMOTEENN helps to address the imbalanced class distribution by both oversampling the minority class and removing noisy samples from both classes.
# 
# 2.Recall_score is a function used to calculate the recall metric for classification tasks. It measures the proportion of actual positive samples that are correctly predicted as positive by a classification model.  This function evaluates the performance of classification models and assess their ability to correctly identify positive samples.
# 
# 3.Classification_report is a function used to generate a text-based report that provides various classification metrics for a classification model. It summarizes the precision, recall, F1-score, and support for each class in a classification task. It evaluates the performance of a classification model and provides insights into its accuracy, precision, recall, and F1-score for each class.
# 
# 4.A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positive, true negative, false positive, and false negative predictions. This function evaluates the performance of classification models and gain insights into the quality of their predictions.
# 
# 5.Decision trees are a type of supervised learning algorithm that can be used for both classification and regression problems. This class provides various parameters and methods that allow you to customize the behavior of the decision tree and fit it to your training data.
# 
# 6.Metrics is a module in scikit-learn that provides various functions for evaluating the performance of machine learning models. It includes metrics for classification, regression, clustering, and model selection. These metrics can be used to assess the performance and effectiveness of your machine learning models on different datasets.
# 
# For example, you can use metrics.accuracy_score to calculate the accuracy of a classification model, metrics.mean_squared_error to calculate the mean squared error of a regression model, or metrics.silhouette_score to evaluate the clustering performance.
# 
# This module allows you to conveniently use these evaluation metrics in your machine learning workflows and assess the quality of your models.

# In[ ]:


df= pd.read_csv('churn.csv')


# In[ ]:


df.head()


# In[ ]:


# remove the unnames column

df = df.drop('Unnamed: 0', axis = 1)


# In[ ]:


df.head()


# In[ ]:


# creating x and y variables. x indepedent variable, and y= dependent variable (Churn)

x= df.drop('Churn', axis = 1)
x


# In[ ]:


# creating x and y variables. x indepedent variable, and y= dependent variable (Churn)

y= df['Churn']
y


# In[2]:


# training and testing

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# **Decision Tree Classifer**

# In[ ]:


model_dt= DecisionTreeClassifier(criterion= 'gini', random_state= 100, max_depth= 6, min_samples_leaf=8)


# criterion='gini': This specifies the criterion used for splitting the tree nodes. In this case, it uses the Gini impurity as the measure of impurity or quality of a split.
# 
# random_state=100: This sets the random seed to ensure reproducibility of the decision tree's results. Setting the random seed to the same value will produce the same results each time the model is run.
# 
# max_depth=6: This sets the maximum depth of the decision tree. It limits the number of levels in the tree. A lower max_depth value can help prevent overfitting and improve generalization.
# 
# min_samples_leaf=8: This sets the minimum number of samples required to be at a leaf node. It prevents the creation of leaf nodes with a small number of samples, which can help avoid overfitting.

# In[ ]:


model_dt.fit(x_train, y_train)


# In[ ]:


#creating the y prediction by calling the model (model_dt) and calling x_test 
#This predicts the target variable values (y) using the trained decision tree model (model_dt) on the test data (x_test).

y_pred = model_dt.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


#calculate the accuracy of the model_dt classifier by comparing the predicted labels (y_pred) with the true labels (x_test).

model_dt.score(x_test, y_pred)


# BREAKDOWN
# 
# model_dt: This refers to the trained decision tree classifier model that has been fit to the training data.
# 
# .score(): This is a method of the DecisionTreeClassifier class in scikit-learn that calculates the accuracy of the model.
# 
# x_test: This represents the input features of the test data on which you want to evaluate the model's accuracy.
# 
# y_pred: This refers to the predicted labels generated by the decision tree model for the test data (x_test).

# In[ ]:


print (classification_report(y_test, y_pred, labels=[0,1]))


# If noticed, the characteristics for the churners which is 1 is not high meaning the accuracy isn't high enough. Therefore we would call the SMOTEENN function to help with the imbalanced dataset

# In[ ]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Calling the SMOTEENN function to help resample the imbalanced data

sm = SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(x,y)


# In[ ]:


xr_train, xr_test, yr_train, yr_test = train_test_split(x_resampled, y_resampled, test_size=0.2)


# In[ ]:


model_dt_smote= DecisionTreeClassifier(criterion= 'gini', random_state= 100, max_depth= 6, min_samples_leaf=8)


# In[ ]:


model_dt_smote.fit(xr_train, yr_train)  #fitting our model


# In[ ]:


y_pred_smote = model_dt_smote.predict(xr_test)


# In[ ]:


print(confusion_matrix(yr_test, y_pred_smote))


# Now, there is a close correlation or accuracy between the TP and FN, also the TN and FP

# In[ ]:


#lets call our classification report to check the accuracy

print (classification_report(yr_test, y_pred_smote, labels=[0,1]))


# This looks a lot more balanced with high accuracy

# Lets try a different Classifier.Building models with different classifiers/algorithms can be seen as a good practiced. Different classifiers are encouraged so one can evaluate and chose which they feel works better with their model.
# 
# **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_rf= RandomForestClassifier(n_estimators= 100, criterion= 'gini', random_state= 100, max_depth= 6, min_samples_leaf=8)
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_test)


# In the RandomForestClassifier, n_estimators is a parameter that specifies the number of decision trees to be created in the random forest. It determines the number of trees to be included in the random forest. Increasing the number of estimators typically improves the performance of the model.

# In[ ]:


print (classification_report(y_test, y_pred_rf, labels=[0,1]))


# In[ ]:


# using smoteenn

sm = SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(x,y)


# In[ ]:


xr_train, xr_test, yr_train, yr_test = train_test_split(x_resampled, y_resampled, test_size=0.2)


# In[ ]:


model_smote_rf= RandomForestClassifier(n_estimators= 100, criterion= 'gini', random_state= 100, max_depth= 6, min_samples_leaf=8)


# In[ ]:


model_smote_rf.fit(xr_train, yr_train)  #fitting our model


# In[ ]:


y_pred_smote_rf = model_smote_rf.predict(xr_test)


# In[ ]:


print(confusion_matrix(yr_test, y_pred_smote_rf))


# In[ ]:


print (classification_report(yr_test, y_pred_smote_rf, labels=[0,1]))  #higher accuracy when smoteenn was used.


# In[ ]:


#save the model

import pickle


# In[3]:


import joblib


# Model Serialization: Serialize or save the trained model into a file format that can be easily loaded and used later. Common serialization formats include pickle, joblib, or TensorFlow's SavedModel format.

# In[4]:


filename= 'churn_model.sav'


# In[ ]:


#dump model

joblib.dump(model_smote_rf, open(filename, 'wb'))


# In[5]:


#to call the model

load_model = joblib.load('churn_model.sav', 'rb')


# In[ ]:


load_model.score(xr_test, yr_test)


# **Deployment**

# Steps for deployment
# 
# Flask Setup: Install Flask, a web framework for Python, to create a web application for hosting your machine learning model.
# 
# Create API Endpoint: Define an API endpoint in your Flask application where the model predictions will be served. This endpoint will receive incoming requests and return the model's predictions.
# 
# Request Processing: In the Flask API endpoint, process the incoming request data, ensuring it is in the correct format and performing any necessary data preprocessing or transformation.
# 
# Model Loading: Load the serialized model into memory. This step typically happens once during the application's startup to avoid reloading the model for each request.
# 
# Prediction: Use the loaded model to make predictions on the processed input data.
# 
# Response Formatting: Format the prediction results into an appropriate response format, such as JSON, and return the response to the client.
# 
# Error Handling: Implement error handling to handle cases such as invalid requests or errors during prediction.
# 
# Deployment: Deploy the Flask application on a server or cloud platform, ensuring it is accessible via an endpoint URL.
# 
# Testing: Test the deployed API by sending sample requests and verifying the responses. Ensure the predictions match the expected results.
# 
# Monitoring and Maintenance: Monitor the performance of the deployed model, track usage metrics, and perform regular maintenance tasks such as updating dependencies or retraining the model as needed.
# 
# With this, you can deploy your machine learning model, allowing it to be accessed and used by other applications or systems over HTTP.
# 
# WHAT IS FLASK?
# 
# In simple terms, Flask is a lightweight micro-web framework for Python that allows you to build web applications. It provides a set of tools and libraries to handle web-related tasks, such as routing incoming requests, processing data, and generating responses.
# 
# Think of Flask as a toolbox that simplifies the process of building a web application. It provides a set of functions and modules that handle common web development tasks, such as handling HTTP requests, rendering HTML templates, and managing sessions.
# 
# WHAT IS API?
# 
# An API (Application Programming Interface) is like a messenger that allows different software applications to communicate and interact with each other. It defines a set of rules and protocols that determine how software components should interact, share data, and request services from each other.
# 
# It is like a contract or agreement between two software systems. It specifies how one system can make requests to another system and what kind of responses it can expect to receive. APIs provide a standardized way for different applications or services to exchange information and perform actions without needing to know the intricate details of each other's internal workings.
# 
# For example, imagine you want to build a weather app. Instead of collecting and maintaining a large database of weather information yourself, you can use a weather API provided by a weather service. The API allows your app to send requests for weather data, such as current temperature or forecast, and the weather service returns the requested information in a structured format that your app can understand and use.
# 

# In[ ]:


import os

print(os.getcwd())


# In[ ]:


"C:\Users\Ezetendu Olive C\Desktop\Back-up"

