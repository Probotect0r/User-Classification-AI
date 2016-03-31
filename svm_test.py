from data import *
from sklearn import svm

data = load_test()

# Learn from the data and then predict
clf = svm.SVC(kernel='linear', C=1).fit(data["training_data"], data["training_targets"])
test_predicted_targets = clf.predict(data["test_data"])
print test_predicted_targets

write_predictions(data["ids"], test_predicted_targets)
