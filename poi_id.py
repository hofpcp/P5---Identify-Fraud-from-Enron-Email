
# coding: utf-8

# In[1]:

## Import all necessary libraries for manipulating data and building the ML models 

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[2]:

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". Added is one new "bonus_salary_multiple"

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'restricted_stock', 'director_fees','to_messages', 
                  'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi', 'bonus_salary_multiple']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    


# In[3]:

### Task 2: Remove outliers. From viewing the relationship between Salary and Bonus "Total" identified and removed.
### Visual inspection of the dataset furthermore revealed a non-person "The Travel Agency in the Park" also removed.

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

data_dict.pop('TOTAL',0)
data_dict.pop('The Travel Agency In the Park',0)


# In[4]:

### Task 3: Create new feature(s) New feature "bonus_salary_multiple" added to dataset and feature list.
### Store to my_dataset for easy export below.


for employee in data_dict:
    if data_dict[employee]['salary'] > 0 and data_dict[employee]['bonus'] > 0:
        data_dict[employee]['bonus_salary_multiple'] = round(float(data_dict[employee]['bonus'])
                                                             /float(data_dict[employee]['salary']),2)
    else:
        data_dict[employee]['bonus_salary_multiple'] = float(0.0)
        
for employee in data_dict:
    for i in features_list:
        if data_dict[employee][i] == 'NaN' or data_dict[employee][i] == 'nan':
            data_dict[employee][i] = 0.0


my_dataset = data_dict


# In[5]:

### Features reduced via SelectKBest printed including scoring per feature
### For reasons I have not been able to identify the new Feature was needed to be run twice (therefore duplicated here)
### In order for SelectKBest to run successfully

for employee in data_dict:
    if data_dict[employee]['salary'] > 0 and data_dict[employee]['bonus'] > 0:
        data_dict[employee]['bonus_salary_multiple'] = round(float(data_dict[employee]['bonus'])
                                                             /float(data_dict[employee]['salary']),2)
    else:
        data_dict[employee]['bonus_salary_multiple'] = float(0.0)


for employee in data_dict:
    for i in features_list:
        if data_dict[employee][i] == 'NaN' or data_dict[employee][i] == 'nan':
            data_dict[employee][i] = 0.0


my_dataset = data_dict


data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

kbest = SelectKBest()
kbest = SelectKBest(k=10)
selected_features = kbest.fit_transform(features,labels)


features_selected=[features_list[i+1] for i in kbest.get_support(indices=True)]
print 'Features selected by SelectKBest:'
print features_selected

print kbest.scores_


# In[6]:

### Features list updated with the SelectKBest and 'poi' added as first element

poi = 'poi'
features_list = [poi] + features_selected
print features_list


# In[7]:

### Extract features and labels from dataset for local testing
### data = featureFormat(my_dataset, features_list, sort_keys = True)
### labels, features = targetFeatureSplit(data)
### features scaled to between 0-1 via the MinMaxScaler


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


### For the initial test of chosen ML Algorithms features/labels split into train/test sets

labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size = 0.3, random_state = 42)


# In[8]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# GaussianNB classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
gaus_sc = clf.score(features_test, labels_test)

print 'GaussianNB Validation'
print "Score: ", gaus_sc
pre_gaus = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Presicion: ",pre_gaus
rec_gaus = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Recall: ",rec_gaus, '\n'


# Support Vector Machine classifier
from sklearn.svm import SVC
clf = SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
svm_sc = clf.score(features_test, labels_test)

print 'SVM Validation'
print "Score: ", svm_sc
pre_svm = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Presicion: ",pre_svm
rec_svm = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Recall: ",rec_svm, '\n'


# RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
rf_sc = clf.score(features_test, labels_test)

print 'RandomForrest Validation'
print "Score: ", rf_sc
pre_rf = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Presicion: ",pre_rf
rec_rf = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Recall: ",rec_rf, '\n'


# In[9]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Further libraries loaded to do Pipeline, FeatureUnion, Gridsearch, PCA and StratifiedShufflesplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA

# Validation of the "Tuned" Classifiers done via a StratifiedShuffleSplit  
ss_split = StratifiedShuffleSplit(labels, 1000, test_size = 0.5, random_state = 42)

for train_idx, test_idx in ss_split: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )    




# In[10]:

### RandomForest with Parameter Tuning

clf = RandomForestClassifier(criterion = 'gini', min_samples_split = 2, max_leaf_nodes = None, max_depth = 5, 
                             min_samples_leaf = 2, random_state = 0, n_estimators = 200)

parameters = {'max_features': [3, 5]}

grid_search_rf = GridSearchCV(clf, parameters, cv = ss_split)
grid_search_rf.fit(features, labels)

clf = grid_search_rf.best_estimator_

clf.fit(features, labels)

print 'RandomForest Validation with Parameter Tuning'
print ('Best Score: {}'.format(grid_search_rf.best_score_))
print('Best parameters: {}'.format(grid_search_rf.best_params_))
print clf

pred_rf= clf.predict(features_test)

pre_rf = precision_score(labels_test, pred_rf, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Precision: ",pre_rf
rec_rf = recall_score(labels_test, pred_rf, labels=None, pos_label=1, average='binary', sample_weight=None)
print "Recall: ",rec_rf


# In[11]:

### Support Vector Machine with Parameter Tuning


from sklearn.decomposition import PCA

skb = SelectKBest(k = 4)

pca = PCA(n_components=5)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", skb)])

# Use combined features to transform dataset:
features_x = combined_features.fit(features, labels).transform(features)

pipeline = Pipeline([("features", combined_features), ("SupportVectorClassifier", SVC(random_state=42))])


param_grid = {
          'SupportVectorClassifier__C': [1, 10, 100, 1000], 
          'SupportVectorClassifier__gamma': [0.001, 0.0001], 
          'SupportVectorClassifier__kernel': ['rbf', 'linear'],
}




gs = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', cv = ss_split)
gs.fit(features, labels)

clf = gs.best_estimator_

clf.fit(features, labels)
pred = clf.predict(features_test)

accuracy = clf.score(features_test, labels_test)
print "accuracy SVC",accuracy

pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec

print clf


# In[12]:

### GaussianNB with Parameter Tuning

clf = GaussianNB()

from sklearn.decomposition import PCA

skb = SelectKBest(k = 4)

pca = PCA(n_components=5)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", skb)])

# Use combined features to transform dataset:
features_x = combined_features.fit(features, labels).transform(features)

pipeline = Pipeline([("features", combined_features), ("NaiveBayes", GaussianNB())])

pa = dict(features__pca__n_components=[2, 3, 4, 5],
                  features__univ_select__k=[1, 2, 3, 4])

gs = GridSearchCV(pipeline, param_grid = pa, scoring = 'f1', cv = ss_split)
gs.fit(features, labels)

clf = gs.best_estimator_

clf.fit(features, labels)
pred = clf.predict(features_test)

accuracy = clf.score(features_test, labels_test)
print "accuracy GaussianNB",accuracy

pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec

print clf


# In[13]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



