from data import *
from sklearn import linear_model

data = load_training(0.4)
# Learn from the data

logreg = linear_model.LogisticRegression(C=1000)
logreg.fit(data["training_data"], data["training_targets"])
test_predicted_targets = logreg.predict(data["test_data"])
print logreg.score(data["test_data"], data["test_targets"])
# Calculate the error
calculate_error(test_predicted_targets, data["test_targets"])
