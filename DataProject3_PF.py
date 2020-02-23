# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:58:38 2020

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
pf = pd.read_csv ("BIG_BANK_plazofijo.csv", sep=',', decimal='.', header = 0, index_col = "row ID")
print(pf)
pf.shape

    # We rename some columns and values
pf = pf.rename(columns={"emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx", "cons.conf.idx":
    "cons_conf_idx", "nr.employed": "nr_employed"})

pf.education = pf.education.replace(to_replace="basic.4y", value="basic_4y")
pf.education = pf.education.replace(to_replace="high.school", value="high_school")
pf.education = pf.education.replace(to_replace="basic.6y", value="basic_6y")
pf.education = pf.education.replace(to_replace="basic.9y", value="basic_9y")
pf.education = pf.education.replace(to_replace="university.degree", value="university_degree")
pf.education = pf.education.replace(to_replace="professional.course", value="professional_course")

pf.job = pf.job.replace(to_replace="admin.", value="admin")

print(pf.dtypes) # We see the data types
print(pf.isnull().sum()) # We check if there are missing values

pf_1 = pf.copy() # We create a new dataframe to work with.

###### DESCRIBE THE VARIABLES ######
    # Age
print(pf_1.age.describe())
age = pf_1.age
pf_1.loc[(pf_1['age']>=(2010-1980)) ,"age_st"]= "X Gen"
pf_1.loc[((pf_1['age']<(2010-1980)) & (pf_1['age']>(2010-1996))) ,"age_st"]= "Millenials"

pf_1 = pf_1.drop('age', axis = 1)

mytable = pd.crosstab(index = pf_1["age_st"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # Job
print(pf_1.job.describe())
mytable = pd.crosstab(index = pf_1["job"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # marital
print(pf_1.marital.describe())
mytable = pd.crosstab(index = pf_1["marital"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # education
print(pf_1.education.describe())
mytable = pd.crosstab(index = pf_1["education"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # default
print(pf_1.default.describe())
mytable = pd.crosstab(index = pf_1["default"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # housing
print(pf_1.housing.describe())
mytable = pd.crosstab(index = pf_1["housing"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # loan
print(pf_1.loan.describe())
mytable = pd.crosstab(index = pf_1["loan"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # contact
print(pf_1.contact.describe())
mytable = pd.crosstab(index = pf_1["contact"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # month
print(pf_1.month.describe())
mytable = pd.crosstab(index = pf_1["month"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # day_of_week
print(pf_1.day_of_week.describe())
mytable = pd.crosstab(index = pf_1["day_of_week"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # duration
print(pf_1.duration.describe())

plt.hist(pf_1.duration, edgecolor = "Black")
plt.show()

    # campaign
print(pf_1.campaign.describe())

plt.hist(pf_1.campaign, edgecolor = "Black")
plt.show()

    # pdays
print(pf_1.pdays.describe())
pf_1.loc[(pf_1['pdays'] == 999) ,"pdays_st"]= "never"
pf_1.loc[(pf_1['pdays'] == 0) ,"pdays_st"]= "no"
pf_1.loc[((pf_1['pdays'] > 0) & (pf_1["pdays"] < 999)) ,"pdays_st"]= "yes"

pf_1 = pf_1.drop('pdays', axis = 1)

mytable = pd.crosstab(index = pf_1["pdays_st"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # previous
print(pf_1.previous.describe())

plt.hist(pf_1.previous, edgecolor = "Black")
plt.show()

    # poutcome
print(pf_1.poutcome.describe())
pf_1['poutcome'] = pf_1['poutcome'].astype(str) # We convert it to categorical type.
print(pf_1.poutcome.dtypes)

mytable = pd.crosstab(index = pf_1["poutcome"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

    # y
print(pf_1.y.describe())
mytable = pd.crosstab(index = pf_1["y"], columns = "count")

        # Percentages
n = mytable.sum()
mytable = (mytable/n)*100
print(mytable)

        # Graphically (Barchart)
plt.bar(mytable.index, mytable["count"], edgecolor = "Black")
plt.show()

print(pf_1.dtypes)

##### MODEL VALIDATION #####
    
    # One-hot encoding
pf_1.loc[(pf_1['y']=='yes') ,"y"]= 1
pf_1.loc[(pf_1['y']=='no') ,"y"]= 0
pf_1 = pd.get_dummies(pf_1)

pf_cluster = pf_1 # For using later in k-means clustering

    # We split the dato into features and target
features = pf_1.loc[:, pf_1.columns != 'y']
target = np.array(pf_1['y'])

    # As the target is unbalanced, in order to make a correct splitting for training the model
    # we need to use cross-validation.
from sklearn.model_selection import train_test_split
perc_values = [0.8, 0.2]

X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(features, target, test_size = perc_values[1], random_state = 1)

print('Train data size = ' + str(X_train_cross.shape))
print('Ttrain target size = ' + str(y_train_cross.shape))
print('Test data size = ' + str(X_test_cross.shape))
print('Test target size = ' + str(y_test_cross.shape))

    # To know which hyperparameters are the best ones, we need to make a grid search.
        # 1. Choose the family models and transform string to float
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

        # 2. Choose hyperparameters to optimize
from sklearn.model_selection import GridSearchCV
            # Create the parameter grid 
param_grid = {
    'bootstrap': [True],
    'max_depth': [30, 40],
    'max_features': [5, 6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [5, 10, 15],
    'n_estimators': [200, 300, 500]
}

            # Instantiate the grid search model
from sklearn.metrics import f1_score, make_scorer
f1 = make_scorer(f1_score, average = "macro")

grid_results_cv = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, n_jobs = -1,  verbose = 3, scoring = f1)
grid_results_cv.fit(X_train_cross, y_train_cross)

print("Best hyperparameters: " + str(grid_results_cv.best_params_)) # Best hyperparameters
#bootstrap = grid_results_cv.best_estimator_.bootstrap #True
#max_depth = grid_results_cv.best_estimator_.max_depth #30
#max_features = grid_results_cv.best_estimator_.max_features #6
#min_samples_leaf = grid_results_cv.best_estimator_.min_samples_leaf #3
#min_samples_split = grid_results_cv.best_estimator_.min_samples_split #5
#n_estimators = grid_results_cv.best_estimator_.n_estimators #200

bootstrap = True
max_depth = 30
max_features = 6
min_samples_leaf = 3
min_samples_split = 5
n_estimators = 200

print("Best score: " + str(grid_results_cv.best_score_))

        # 3. Train the model
model = RandomForestClassifier(bootstrap = bootstrap, max_depth = max_depth, max_features = max_features, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, n_estimators = n_estimators)
model.fit(X_train_cross, y_train_cross)

        # 4. Generate the predictions
pred_train = model.predict(X_train_cross)
pred_test = model.predict(X_test_cross)

        # 5. Calculate metrics
from sklearn.metrics import accuracy_score as acc

acc_train = acc(y_train_cross, pred_train)
acc_test = acc(y_test_cross, pred_test)

print('accuracy train = ' + str(acc_train))
print('accuracy test = ' + str(acc_test))

        # Confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test_cross, pred_test)

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
pred_train_p = model.predict_proba(X_train_cross)
pred_test_p = model.predict_proba(X_test_cross) 

            # Calculate the evaluation metrics AUC
from sklearn.metrics import roc_auc_score as auc

auc_train = auc(y_train_cross, pred_train_p[:,1]);
auc_test = auc(y_test_cross, pred_test_p[:,1]);
results = pd.DataFrame()
results = results.append(pd.DataFrame(data={'auc_train':[auc_train],'auc_test':[auc_test]}, columns=['auc_train', 'auc_test']), ignore_index=True)

print(results)

            # Visualize the ROC curve
import scikitplot as skplt

skplt.metrics.plot_roc(y_test_cross, pred_test_p, plot_macro = False, plot_micro = False, classes_to_plot = 1)

        # 7. Cumulative gain
import scikitplot as skplt

skplt.metrics.plot_cumulative_gain(y_test_cross, pred_test_p)

        # 8. Lift curve
import scikitplot as skplt
skplt.metrics.plot_lift_curve(y_test_cross, pred_test_p)

# VISUALIZE THE RANDOM FOREST
    # Versión original
from sklearn.tree import export_graphviz
import pydotplus

clf = RandomForestClassifier(bootstrap = bootstrap, max_depth = max_depth, max_features = max_features, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, n_estimators = n_estimators)
clf.fit(X_train_cross, y_train_cross)


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
clf.fit(X_train_cross, y_train_cross)


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
                                   index = X_train_cross.columns,
                                    columns=['importance'])
feature_importances.sort_values('importance', ascending=False, inplace = True)
feature_importances = feature_importances.iloc[0:9]

plt.barh(feature_importances.index, feature_importances.importance, edgecolor = "Black")
plt.gca().invert_yaxis()
plt.show()

# 1.1 CLUSTERING
from sklearn.preprocessing import scale

pf_cluster = pf_cluster.loc[pf_cluster["age_st_Millenials"]== 1] # We select only millenials

y_cluster = pd.DataFrame(pf_cluster["y"])
y_cluster = y_cluster.reset_index(drop = True)

pf_cluster = pf_cluster.drop(["y"], axis = 1)

# K-means
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(pf_cluster)
    distortions.append(kmeanModel.inertia_)

# Let´s plot our result
plt.figure(figsize=(6,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Train the K-means model
kmeanModel = KMeans(n_clusters=3)
kmeanModel.fit(pf_cluster)
y_kmeans = kmeanModel.predict(pf_cluster)
centers = kmeanModel.cluster_centers_

kmeans_df = pd.DataFrame(centers)
kmeans_df.columns = pf_cluster.columns

pf_pca = pd.DataFrame(scale(pf_cluster))
pf_pca.columns = pf_cluster.columns

pf_pca = pf_pca[["duration", "campaign"]]

plt.scatter(pf_pca.duration, pf_pca.campaign, alpha=0.9, c=y_kmeans, cmap="brg")
plt.xlabel('duration')
plt.ylabel('campaign')

# Compare our original data versus our clustered results
new_labels = pd.DataFrame(kmeanModel.labels_)

pf_cluster = pf_cluster.reset_index(drop = False)

pf_cluster = pf_cluster.join(new_labels)
pf_cluster = pf_cluster.join(y_cluster)
pf_cluster = pf_cluster.rename(columns={0: "cluster"})

# CORRELACIÓN CLUSTER Y VENTAS
import scipy.stats as stats
pf_cluster.loc[(pf_cluster['cluster'] == 0) ,"cluster"]= "cluster 0"
pf_cluster.loc[(pf_cluster['cluster'] == 1) ,"cluster"]= "cluster 1"
pf_cluster.loc[(pf_cluster['cluster'] == 2) ,"cluster"]= "cluster 2"

pf_cluster.loc[(pf_cluster['y'] == 0) ,"y"]= "no"
pf_cluster.loc[(pf_cluster['y'] == 1) ,"y"]= "yes"

my_ct = pd.crosstab(pf_cluster.y, pf_cluster.cluster, normalize = "columns", margins = True)*100
my_ct = round(my_ct, 1)
print(my_ct)
  # Perform the Chi2 test (p value < 0.05) of the single crosstab.  It gives us
  # a matrix of hoped frequencies.
res = stats.chi2_contingency(my_ct)
print(res)

Chi2 = res[0]
P_val = res[1]

 # Graphical comparison.  We need to transpose the crosstab and then plot
my_ct.T.plot(kind = "Bar", edgecolor = "Black", colormap = "Paired",)
plt.ylim(0, 110)
props = dict(boxstyle = "round", facecolor = "white", lw = 0.5)
xmin, xmax, ymin, ymax = plt.axis()
plt.xlabel("Clusters")

# We define characteristics of cluster 1
pf_mil = pf[((pf['age']<(2010-1980)) & (pf['age']>(2010-1996)))]
pf_mil = pf_mil.reset_index(drop = True)
new_labels = new_labels.rename(columns={0: "cluster"})
pf_mil["cluster"] = new_labels["cluster"]

pf_mil_c0 = pf_mil[pf_mil["cluster"] == 0]
pf_mil_c1 = pf_mil[pf_mil["cluster"] == 1]
pf_mil_c2 = pf_mil[pf_mil["cluster"] == 2]

means_c1 = pd.DataFrame(pf_mil_c1.describe())
pf_mil_c1.job.mode(0)
pf_mil_c1.marital.mode(0)
pf_mil_c1.education.mode(0)
pf_mil_c1.housing.mode(0)
pf_mil_c1.loan.mode(0)
pf_mil_c1.contact.mode(0)
pf_mil_c1.month.mode(0)
