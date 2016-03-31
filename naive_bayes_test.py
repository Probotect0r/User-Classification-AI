from data import *
from sklearn.naive_bayes import GaussianNB

data = load_test()

# Learn from the data
gnb = GaussianNB().fit(data["training_data"], data["training_targets"])
test_predicted_targets = gnb.predict(data["test_data"])
print test_predicted_targets

write_predictions(data["ids"], test_predicted_targets)
