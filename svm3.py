# import json
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.multiclass import OneVsRestClassifier

# JSON_PATH = "data.json"

# # Load in the pre-processed data from our JSON file
# def load_data(json_path):
#     with open(json_path, "r") as input:
#         data = json.load(input)

#     # Split the genres and their labels into NumPy arrays 
#     X = np.array(data["mfccs"])
#     y = np.array(data["labels"])

#     return X, y

# # We want to apply one-hot-encoding to our labels to help our model understand the genre labelling better
# def one_hot_encoding(y):
#     # Initialize the one-hot-encoder
#     encoder = OneHotEncoder(sparse_output=False)
#     # Reshape the label data into a 2-D vector before performing the encoding
#     y_encoded = encoder.fit_transform(y.reshape(-1, 1))
#     return y_encoded

# # Split the data into training and testing sets
# def split_data(X, y, test_size=0.2, random_state=42):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#     # The X_train and X_test currently have three dimensions but the SVM only works with two -> so we need to flatten the layers
#     X_train_flat = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
#     X_test_flat = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])

#     return X_train_flat, X_test_flat, y_train, y_test


# # Train the SVM model
# def trainSVM(X_train, y_train):
#     # Initialize the Support Vector Classifier setting our hyperparameters that we will potentially adjust later 
#     svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
#     ovr_model = OneVsRestClassifier(svm_model)
#     ovr_model.fit(X_train, y_train)
#     return ovr_model

# def evaluate_model(model, X_test, y_test):
#     y_test1d = np.argmax(y_test, axis=1)
#     accuracy = model.score(X_test, y_test1d)
#     print("Accuracy: {:.2f}".format(accuracy * 100))

# # Step 1: Load the JSON data and extract features and labels
# DATASET_PATH = "data.json"
# X, y = load_data(DATASET_PATH)

# # Step 2: Apply one-hot encoding for multi-class classification
# y = one_hot_encoding(y)

# # Step 3: Split the data into training and testing sets
# X_train_flat, X_test_flat, y_train, y_test = split_data(X, y)

# # Step 4: Train SVM model
# ovr_model = trainSVM(X_train_flat, y_train)

# # Step 5: Evaluate the model's performance
# evaluate_model(ovr_model, X_test_flat, y_test)



# import json
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# JSON_PATH = "data.json"

# # Load in the pre-processed data from our JSON file
# def load_data(json_path):
#     with open(json_path, "r") as input_file:
#         data = json.load(input_file)

#     X = np.array(data["mfccs"])
#     y = np.array(data["labels"])

#     print("shape of X", X.shape)
#     print("shape of y", y.shape)

#     return X, y

# if __name__ == "__main__":
#     X, y = load_data(JSON_PATH)

# # Split the data into training and testing sets
# def split_data(X, y, test_size=0.2, random_state=42):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#     # Reshape the feature data to be 2D for SVM
#     X_train_flat = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
#     X_test_flat = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])

#     # Reshape the label data to be 1D for SVM
#     y_train = y_train.ravel()
#     y_test = y_test.ravel()

#     return X_train_flat, X_test_flat, y_train, y_test


# # Split the data into training and testing sets
# def split_data(X, y, test_size=0.2):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

#     print("X_train shape after splitting:", X_train.shape)
#     print("y_train shape after splitting:", y_train.shape)

#     # Merge the segments for training data
#     X_train_merged = np.concatenate(X_train, axis=0)

#     # Merge the segments for testing data
#     X_test_merged = np.concatenate(X_test, axis=0)

#     # Flatten the y_train array to 1D
#     y_train = y_train.ravel()  # Convert to 1D array

#     print("X_train_merged shape after merging:", X_train_merged.shape)
#     print("y_train shape after flattening:", y_train.shape)

#     # Verify if the number of samples in X_train_merged and y_train match
#     assert X_train_merged.shape[0] == y_train.shape[0]

#     return X_train_merged, X_test_merged, y_train, y_test

# if __name__ == "__main__":
#     X_train_merged, X_test_merged, y_train, y_test = split_data(X, y)

# # Train the SVM model
# def trainSVM(X_train, y_train):
#     # Initialize the Support Vector Classifier setting our hyperparameters that we will potentially adjust later
#     svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
#     ovr_model = OneVsRestClassifier(svm_model)
#     ovr_model.fit(X_train, y_train)
#     return ovr_model


# def evaluate_model(model, X_test, y_test):
#     accuracy = model.score(X_test, y_test)
#     print("Accuracy: {:.2f}".format(accuracy * 100))

# if __name__ == "__main__":
#     X, y = load_data(JSON_PATH)
#     X_train_merged, X_test_merged, y_train, y_test = split_data(X, y)
#     svm_model = trainSVM(X_train_merged, y_train)
#     evaluate_model(svm_model, X_test_merged, y_test)

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

JSON_PATH = "data.json"

# Load in the pre-processed data from our JSON file
def load_data(json_path):
    with open(json_path, "r") as input:
        data = json.load(input)

    X = np.array(data["mfccs"])
    y = np.array(data["labels"])

    print("mfccs", X[0:5])
    print("labels", y[0:5])

    return X, y

def split_data(X, y, test_size=0.9, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    X_train_flat = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
    y_train_flat = np.repeat(y_train, X_train.shape[1])
    print("X_train shape:", X_train_flat.shape)
    print("y_train shape:", y_train_flat.shape)
    return X_train_flat, X_test, y_train_flat, y_test

def trainSVM(X_train, y_train):
    svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print("Accuracy: {:.2f}".format(accuracy * 100))

DATASET_PATH = "data.json"
X, y = load_data(DATASET_PATH)

X_train, X_test, y_train, y_test = split_data(X, y)

svm_model = trainSVM(X_train, y_train)

evaluate_model(svm_model, X_test, y_test)


