# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:29:14 2020

@author: ncosn
"""

#load basiclibraries
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# Get working directory
os.getcwd()

# Change working directory
os.chdir("C:/Users/ncosn/Downloads")
os.getcwd()

# Reads data from CSV file and stores it in a dataframe called pf (plazo fijo)
# Pay atention to the specific format of your CSV data (; , or , .)
prest = pd.read_csv ("BIG_BANK_prestamo.csv", sep=',', decimal='.', header = 0)
print(prest)
prest.shape

##### PREPROCESSING #####
    # Eliminate duplicates and non-interesting variables
prest = prest.drop_duplicates("Loan ID")
prest = prest.drop(["row ID", "Loan ID", "Customer ID"], axis = 1)

print(prest.dtypes) # We see the data types
print(prest.isnull().sum()) # We check if there are missing values

    # Correct some values
prest.loc[prest["Current Loan Amount"] == 99999999, "Loan Status"] = "Fully Paid" # Relation. All 99999999 are Fully Paid.
prest.loc[prest["Credit Score"] >= 1000, "Credit Score"] = prest["Credit Score"]/10 # Credit score [300, 850]
prest.loc[prest["Home Ownership"] == "HaveMortgage", "Home Ownership"] = "Home Mortgage"
prest.loc[prest["Purpose"] == "other", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "Take a Trip", "Purpose"] = "vacation"

prest = prest.drop(["Months since last delinquent"], axis = 1) # Eliminate; >50% values = missing values

# DESCRIBE THE VARIABLES TO FIND OUTLIERS. IF NOT, WE FILL MISSING DATA WITH MODE (CAT) AND MEAN (NUM)
    # Loan Status
print(prest["Loan Status"].describe())

mytable = pd.crosstab(index = prest["Loan Status"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

prest["Loan Status"].fillna(prest["Loan Status"].mode()[0], inplace=True)

    # Current Loan Amount
print(prest["Current Loan Amount"].describe())

plt.hist(prest["Current Loan Amount"], edgecolor = "Black")
plt.show()

    # Term
print(prest["Term"].describe())

mytable = pd.crosstab(index = prest["Term"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # Credit Score
print(prest["Credit Score"].describe())

prest["Credit Score"].fillna(prest["Credit Score"].mean(), inplace=True)

plt.hist(prest["Credit Score"], edgecolor = "Black")
plt.show()

    # Annual Income
print(prest["Annual Income"].describe())

plt.hist(prest["Annual Income"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Annual Income"].dropna())

na = prest[pd.isnull(prest["Annual Income"])] # Remove NaN values and save them
prest = prest[prest["Annual Income"]<=1.5e8] # Remove outliers
prest = prest.append(na) # Append NaN values to replacing them by the mean

prest["Annual Income"].fillna(prest["Annual Income"].mean(), inplace=True)

    # Years in current job
print(prest["Years in current job"].describe())

mytable = pd.crosstab(index = prest["Years in current job"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

prest["Years in current job"].fillna(prest["Years in current job"].mode()[0], inplace=True)

    # Home Ownership
print(prest["Home Ownership"].describe())

mytable = pd.crosstab(index = prest["Home Ownership"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # Purpose
print(prest["Purpose"].describe())

mytable = pd.crosstab(index = prest["Purpose"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

        # We include in category "Other" every category with less than a 1% of presence
prest.loc[prest["Purpose"] == "Buy House", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "Educational Expenses", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "major_purchase", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "moving", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "renewable_energy", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "small_business", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "vacation", "Purpose"] = "Other"
prest.loc[prest["Purpose"] == "wedding", "Purpose"] = "Other"

    # Monthly Debt
print(prest["Monthly Debt"].describe())

plt.hist(prest["Monthly Debt"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Monthly Debt"].dropna())

prest = prest[prest["Monthly Debt"]<=400000] # Remove outliers

    # Years of Credit History
print(prest["Years of Credit History"].describe())

plt.hist(prest["Years of Credit History"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Years of Credit History"].dropna())

    # Number of Open Accounts
print(prest["Number of Open Accounts"].describe())

plt.hist(prest["Number of Open Accounts"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Number of Open Accounts"].dropna())

    # Number of Credit Problems
print(prest["Number of Credit Problems"].describe())

plt.hist(prest["Number of Credit Problems"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Number of Credit Problems"].dropna())

    # Current Credit Balance
print(prest["Current Credit Balance"].describe())

plt.hist(prest["Current Credit Balance"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Current Credit Balance"].dropna())

prest = prest[prest["Current Credit Balance"]<=3e7] # Remove outliers

    # Months since last delinquent
print(prest["Maximum Open Credit"].describe())

plt.hist(prest["Maximum Open Credit"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Maximum Open Credit"].dropna())

prest["Maximum Open Credit"].fillna(prest["Maximum Open Credit"].mean(), inplace=True)

    # Bankruptcies
print(prest["Bankruptcies"].describe())

plt.hist(prest["Bankruptcies"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Bankruptcies"].dropna())

prest["Bankruptcies"].fillna(prest["Bankruptcies"].mean(), inplace=True)

    # Tax Liens
print(prest["Tax Liens"].describe())

plt.hist(prest["Tax Liens"], edgecolor = "Black")
plt.show()

plt.boxplot(prest["Tax Liens"].dropna())

prest["Tax Liens"].fillna(prest["Tax Liens"].mean(), inplace=True)

print(prest.isnull().sum()) # No missing values

##### MODEL VALIDATION #####
   
prest1 = prest.copy()

    # One-hot encoding
prest1.loc[(prest1['Loan Status']=='Fully Paid') ,"Loan Status"]= 1
prest1.loc[(prest1['Loan Status']=='Charged Off') ,"Loan Status"]= 0
prest1 = pd.get_dummies(prest1)

    # We split the dato into features and target
features = prest1.loc[:, prest1.columns != 'Loan Status']
target = np.array(prest1['Loan Status'])

    # As the target is balanced, we can split the data into train, validation and test.
from sklearn.model_selection import train_test_split
perc_values = [0.7, 0.15, 0.15]

X_train_rand, X_valtest_rand, y_train_rand, y_valtest_rand = train_test_split(features, target, test_size=perc_values[1] + perc_values[2], random_state=1);

X_val_rand, X_test_rand, y_val_rand, y_test_rand = train_test_split(X_valtest_rand, y_valtest_rand, test_size= perc_values[2] / (perc_values[1] + perc_values[2]), random_state=1)

print('Train data size = ' + str(X_train_rand.shape))
print('Train target size = ' + str(y_train_rand.shape))
print('Validation data size = ' + str(X_val_rand.shape))
print('Validation target size = ' + str(y_val_rand.shape))
print('Test data size = ' + str(X_test_rand.shape))
print('Test target size = ' + str(y_test_rand.shape))

    # To know which are the best hyperparameters we need to make a grid search.
        # 1. Choose the family models and transform string to float
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

        # 2. Choose hyperparameters to optimize
from sklearn.model_selection import GridSearchCV
            # Create the parameter grid 
param_grid = {
    'max_depth': [80, 90, 100],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [2, 3],
    'n_estimators': [50, 100, 200]
}

            # Instantiate the grid search model
from sklearn.metrics import f1_score, make_scorer
f1 = make_scorer(f1_score, average = "macro")

grid_results_cv = GridSearchCV(estimator = model, param_grid = param_grid, cv = 4, n_jobs = -1,  verbose = 3, scoring = f1)
grid_results_cv.fit(X_train_rand, y_train_rand)

print("Best hyperparameters: " + str(grid_results_cv.best_params_)) # Best hyperparameters

max_depth = grid_results_cv.best_estimator_.max_depth #40
max_features = grid_results_cv.best_estimator_.max_features
min_samples_leaf = grid_results_cv.best_estimator_.min_samples_leaf #2
min_samples_split = grid_results_cv.best_estimator_.min_samples_split #5
n_estimators = grid_results_cv.best_estimator_.n_estimators #100

bootstrap = True
max_depth = 90
max_features = 4
min_samples_leaf = 3
min_samples_split = 3
n_estimators = 50

print("Best score: " + str(grid_results_cv.best_score_))

        # 3. Train the model
model = RandomForestClassifier(max_depth = max_depth, max_features = max_features, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, n_estimators = n_estimators)
model.fit(X_train_rand, y_train_rand)

        # 4. Generate the predictions
pred_train = model.predict(X_train_rand)
pred_val = model.predict(X_val_rand)
pred_test = model.predict(X_test_rand)

        # 5. Calculate metrics
from sklearn.metrics import accuracy_score as acc

acc_train = acc(y_train_rand, pred_train)
acc_val = acc(y_val_rand, pred_val)
acc_test = acc(y_test_rand, pred_test)

print('accuracy train = ' + str(acc_train))
print('accuracy val = ' + str(acc_val))
print('accuracy test = ' + str(acc_test))

        # 6. Confusion matrix
        
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test_rand, pred_test)

prec = cnf_matrix[1,1] / (cnf_matrix[0,1] + cnf_matrix[1,1])
print("The precision of the ML model is ", round(prec, 3))

recl = cnf_matrix[1,1] / (cnf_matrix[1,0] + cnf_matrix[1,1])
print("The recall of the ML model is ", round(recl, 3))

f1 = 2*((recl*prec)/(recl+prec))
print("The f1-score of the ML model is %f." % round(f1, 3))

acc = (cnf_matrix[1,1] + cnf_matrix[0,0]) / ((cnf_matrix[0,1] + cnf_matrix[1,1]) + cnf_matrix[0,0] + cnf_matrix[1,0])
print("The accuracy of the ML model is ", round(acc, 3))

        # 6. ROC Curve
            # Calculate the probabilities of being 0 or 1
pred_train_p = model.predict_proba(X_train_rand)
pred_val_p = model.predict_proba(X_val_rand)
pred_test_p = model.predict_proba(X_test_rand) 

            # Calculate the evaluation metrics AUC
from sklearn.metrics import roc_auc_score as auc

auc_train = auc(y_train_rand, pred_train_p[:,1])
auc_val = auc(y_val_rand, pred_val_p[:,1])
auc_test = auc(y_test_rand, pred_test_p[:,1])
results = pd.DataFrame()
results = results.append(pd.DataFrame(data={'auc_train':[auc_train], 'auc_val':[auc_val],'auc_test':[auc_test]}, columns=['auc_train', 'auc_val', 'auc_test']), ignore_index=True)

print(results)

            # Visualize the ROC curve
import scikitplot as skplt

skplt.metrics.plot_roc(y_test_rand, pred_test_p, plot_macro = False, plot_micro = False, classes_to_plot = 1)

        # 7. Cumulative gain
import scikitplot as skplt

skplt.metrics.plot_cumulative_gain(y_test_rand, pred_test_p)

        # 8. Lift curve
import scikitplot as skplt
skplt.metrics.plot_lift_curve(y_test_rand, pred_test_p)

# VISUALIZE THE RANDOM FOREST
    # Versión original
from sklearn.tree import export_graphviz
import pydotplus

clf = RandomForestClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, n_estimators = n_estimators)
clf.fit(X_train_rand, y_train_rand)


        # Export as dot file
dot_data = export_graphviz(clf.estimators_[5], out_file=None, 
                feature_names = list(features.columns),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")

        # Display the tree
from IPython.display import Image
Image(filename = 'tree.png')

    # Versión reducida
clf = RandomForestClassifier(max_depth = 3)
clf.fit(X_train_rand, y_train_rand)


        # Export as dot file
dot_data = export_graphviz(clf.estimators_[5], out_file=None, 
                feature_names = list(features.columns),
                class_names = ["yes", "no"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree2.png")

        # Display the tree
from IPython.display import Image
Image(filename = 'tree2.png')

# FEATURE IMPORTANCE
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train_rand.columns,
                                    columns=['importance'])
feature_importances.sort_values('importance', ascending=False, inplace = True)
feature_importances = feature_importances.iloc[0:9]

plt.barh(feature_importances.index, feature_importances.importance, edgecolor = "Black")
plt.gca().invert_yaxis()
plt.show()