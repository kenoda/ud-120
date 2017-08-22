#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import metrics

clf = DecisionTreeClassifier()
feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state = 42)


clf.fit(feature_train, label_train)
acc = clf.score(feature_test, label_test)
print "Accuracy:", acc

pred = clf.predict(feature_test)
print "numbers of poi predicted in test:", int(sum(pred))
print "numbers of ppl in test:", len(feature_test)

for pred, labels in zip(pred, label_test):
    if pred != labels:
        print "they are different"

print "label_test:", label_test
print "pred:", pred

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_positive_counter = true_negative_counter = false_positive_counter= false_negative_counter = 0
for pred, labels in zip(predictions, true_labels):
    if pred == labels:
        if pred == 1:
            true_positive_counter += 1
        else:
            true_negative_counter += 1
    else:
        if pred == 1:
            false_positive_counter += 1
        else:
            false_negative_counter += 1

print "numbers of true positives:", true_positive_counter
print "numbers of true negatives:", true_negative_counter
print "numbers of false positives:", false_positive_counter
print "numbers of false negatives:", false_negative_counter
print "precision:", metrics.precision_score(true_labels, predictions)
print "recall:", metrics.recall_score(true_labels, predictions)

# pred_proba = clf.predict_proba(feature_test)
# print "pred proba:", pred_proba
# print "precision:", metrics.precision_score(label_test, pred_proba)
# print "recall:", metrics.recall_score(label_test, pred)