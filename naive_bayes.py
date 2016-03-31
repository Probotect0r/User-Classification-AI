from data import *
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

data = load_training(0.4)

# Learn from the data
gnb = GaussianNB().fit(data["training_data"], data["training_targets"])
test_predicted_targets = gnb.predict(data["test_data"])
print gnb.score(data["test_data"], data["test_targets"])

# Calculate the error
calculate_error(test_predicted_targets, data["test_targets"])
