from data import *
from sklearn import svm
from sklearn import cross_validation

data = load_training(0.4)
# Learn from the data
clf = svm.SVC(kernel='linear', C=1).fit(data["training_data"], data["training_targets"])
test_predicted_targets = clf.predict(data["test_data"])
print clf.score(data["test_data"], data["test_targets"])

print test_predicted_targets

# Calculate the error
calculate_error(test_predicted_targets, data["test_targets"])
