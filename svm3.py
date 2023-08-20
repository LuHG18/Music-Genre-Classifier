import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

JSON_PATH = "data.json"

# Load in the pre-processed data from our JSON file
def load_data(json_path):
    with open(json_path, "r") as input:
        data = json.load(input)

    X = np.array(data["mfccs"]) # load mfcc features
    y = np.array(data["labels"]) # load corresponding labels

    # check for the correct data being loaded in
    print("mfccs", X[0:5])
    print("labels", y[0:5])

    return X, y

def split_data(X, y, test_size=0.75, random_state=42):
    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # check the shape of the split to ensure that everything matches up
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # flatten the data so that it fits into the proper dimensions for the SVM model
    X_train_flat = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
    y_train_flat = np.repeat(y_train, X_train.shape[1])

    # check the shape of the flattened data
    print("X_train_flat shape:", X_train_flat.shape)
    print("y_train_flat shape:", y_train_flat.shape)

    X_test_flat = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])
    y_test_flat = np.repeat(y_test, X_test.shape[1])

    return X_train_flat, X_test_flat, y_train_flat, y_test_flat

def trainSVM(X_train, y_train):
    svm_model = SVC(kernel='rbf', C=1.0, gamma='auto') # initialize the SVM model
    svm_model.fit(X_train, y_train) # train the model
    return svm_model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print("Accuracy: {:.2f}".format(accuracy * 100))

DATASET_PATH = "data.json"
X, y = load_data(DATASET_PATH)

X_train, X_test, y_train, y_test = split_data(X, y)

svm_model = trainSVM(X_train, y_train)

evaluate_model(svm_model, X_test, y_test)


