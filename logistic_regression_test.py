from data import *
from sklearn import linear_model

data = load_test()

# Learn from the data and then predict
logreg = linear_model.LogisticRegression(C=1000)
logreg.fit(data["training_data"], data["training_targets"])
test_predicted_targets = logreg.predict(data["test_data"])

write_predictions(data["ids"], test_predicted_targets)
