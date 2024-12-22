# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:36:52 2024

@author: kenneyke
"""
#%% Import libraries
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
#import graphviz
import matplotlib.pyplot as plt

#%% Function definitions
def reclassify_labels(X_train, y_train, X_test, y_test):
   # Specify the reclassification label
   reclassification_label = input("Enter the reclassification label: ")

   # Ensure y_train and y_test are pandas Series
   y_train = pd.Series(y_train)
   y_test = pd.Series(y_test)

   unique_labels_train = set(y_train)
   unique_labels_test = set(y_test)

   common_labels = unique_labels_train.intersection(unique_labels_test)
   removed_values_train = unique_labels_train - common_labels
   removed_values_test = unique_labels_test - common_labels
   
   print("Reclassified non-common values from y_train:")
   for label in removed_values_train:
       count = sum(y_train == label)
       print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

   print("\nReclassified non-common values from y_test:")
   for label in removed_values_test:
       count = sum(y_test == label)
       print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

   # Reclassify labels not present in both y_train and y_test
   y_train[~y_train.isin(common_labels)] = reclassification_label
   y_test[~y_test.isin(common_labels)] = reclassification_label
   return X_train, y_train, X_test, y_test
   

def synchronize_labels(X_train, y_train, X_test, y_test, excel_file_path):
    # Ensure labels in y_test are present in y_train, and vice versa
    unique_labels_train = set(y_train)
    unique_labels_test = set(y_test)

    common_labels = unique_labels_train.intersection(unique_labels_test)

    # Find and print removed values
    removed_values_train = set(y_train) - common_labels
    removed_values_test = set(y_test) - common_labels
    
    # Load Excel file
    excel_data = pd.read_excel(excel_file_path)
    ext_class_labels_mapping = dict(zip(excel_data['Ext_Class'], excel_data['Labels']))
    
    print("Removed values from y_train:")
    for label in removed_values_train:
        count = sum(y_train == label)
        ext_class_label = ext_class_labels_mapping.get(label, 'Not Found')
        print(f"{label} ({ext_class_label}): {count} occurrences")

    print("\nRemoved values from y_test:")
    for label in removed_values_test:
        count = sum(y_test == label)
        ext_class_label = ext_class_labels_mapping.get(label, 'Not Found')
        print(f"{label} ({ext_class_label}): {count} occurrences")

    # Remove rows with labels not present in both y_train and y_test
    mask_train = y_train.isin(common_labels)
    mask_test = y_test.isin(common_labels)

    X_train_synchronized = X_train[mask_train]
    y_train_synchronized = y_train[mask_train]

    X_test_synchronized = X_test[mask_test]
    y_test_synchronized = y_test[mask_test]

    return X_train_synchronized, y_train_synchronized, X_test_synchronized, y_test_synchronized

#%% Load data and perform Classification

# # Load CSV data
# data = pd.read_csv(r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\ind_objects_data.csv')

# # Specify the X data & target Data Y
# cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_Class %', 'Total Points', 'Ext_Class', 'Root_Class', 'Sub_Class']   #Columns to exclude from the X data
# X = data.drop(columns=cols_to_remove, axis=1)    
# y = data['Root_Class']

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # 20% for testing. Random state for split reproducibility

# # Create and fit the Random Forest model
# #RF = RandomForestClassifier()      # Uncomment for hyperparameter tunning
# RF = RandomForestClassifier(n_estimators=100, random_state=42)  #Comment for hyperparam tunning
# RF.fit(X_train, y_train)    #Comment for hyperparam tunning

# # Make predictions on the test set
# predictions = RF.predict(X_test)                #Comment for hyperparam tunning

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy}')

# # Additional evaluation metrics
# print('\nClassification Report:')
# print(classification_report(y_test, predictions))

# # Additional evaluation metrics
# print('\nConfusion Matrix:')
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF.classes_)
# disp.plot()
# plt.show()

#%%  Visuallizing the Results

# #Export the first three decision trees from the forest
# for i in range(3):
#     tree = RF.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                 feature_names=X_train.columns,  
#                                 filled=True,  
#                                 max_depth=2, 
#                                 impurity=False, 
#                                 proportion=True)
#     graph = graphviz.Source(dot_data)
    
#     # Save the decision tree as an image file
#     image_path = f'tree_{i+1}.png'
#     graph.render(filename=image_path, format='png', cleanup=True)
    
#     # Open the saved image file
#     Image(filename=image_path)

#%%  Hyperparameter Tuniing
# param_dist = {'n_estimators': randint(50,500),
#               'max_depth': randint(1,20)}

# # Create a random forest classifier
# RF = RandomForestClassifier()

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(RF, 
#                                   param_distributions = param_dist, 
#                                   n_iter=5, 
#                                   cv=5)

# # Fit the random search object to the data
# rand_search.fit(X_train, y_train)

# # Create a variable for the best model
# best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)

# # Generate predictions with the best model
# y_pred = best_rf.predict(X_test)

# # Evaluate the best model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

# # Additional evaluation metrics
# print('\nClassification Report:')
# print(classification_report(y_test, y_pred))

# # Additional evaluation metrics
# print('\nConfusion Matrix:')
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
# disp.plot()
# plt.show()

#%% Assign train and text data and perform Classification

# Labels file path
labels_file_path = r'D:\ODOT_SPR866\My Label Data Work\Sample Label data for testing\Ext_Class_labels.xlsx'

# Load CSV data
Train_data = pd.read_csv(r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\4_ind_objects\1_T01_HWY068_SB_20200603_173632 Profiler.zfs_18_manual_label_data.csv')
Test_data = pd.read_csv(r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\4_ind_objects\3_T01_HWY068_SB_20200603_173632 Profiler.zfs_30_manual_label_data.csv')

# Training
Train_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class']   #Columns to exclude from the X data
X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)    
y_train = Train_data['Ext_class']

# Testing
Test_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class']   #Columns to exclude from the X data
X_test = Test_data.drop(columns=Test_cols_to_remove, axis=1)    
y_test = Test_data['Ext_class']

# Synchronize or reclassify labels
X_train, y_train, X_test, y_test = synchronize_labels(X_train, y_train, X_test, y_test, labels_file_path)



# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # 20% for testing. Random state for split reproducibility

###### Create and fit the Random Forest model
RF = RandomForestClassifier(n_estimators=100)  #Comment for hyperparam tunning
RF.fit(X_train, y_train)    #Comment for hyperparam tunning

# ######### Create a best classifier
# param_dist = {'n_estimators': randint(50,500),
#               'max_depth': randint(1,20)}

# # Create a random forest classifier
# RF = RandomForestClassifier()

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(RF, 
#                                   param_distributions = param_dist, 
#                                   n_iter=5, 
#                                   cv=5)

# # Fit the random search object to the data
# rand_search.fit(X_train, y_train)

# # Create a variable for the best model
# best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)

# Make predictions on the test set
predictions = RF.predict(X_test)                #Comment for hyperparam tunning

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'\n\nAccuracy: {accuracy}')

# Additional evaluation metrics
print('\nClassification Report:')
print(classification_report(y_test, predictions))

# Additional evaluation metrics
print('\nConfusion Matrix:')
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF.classes_)
disp.plot()
plt.show()
