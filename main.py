import pandas as pd
import numpy as np
import pickle

# Read the data from the data file
dataset = pd.read_csv('iris.data')

# Split the data into X and y
# X contains the set of independent variables
# y contains the dependent variable
X = np.array(dataset.iloc[:, 0:4])
y = np.array(dataset.iloc[:, 4:])

# Label Encoding the data
# It is necessary because y contains different names of flowers, which we have to predict
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = y.ravel()
y = le.fit_transform(y)

# Split the dataset to test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Using Support Vector Classifier for our prediction
from sklearn.svm import SVC
classifier = SVC(kernel='linear').fit(X_train, y_train)

# Create the pickle file
pickle.dump(classifier, open('iris.pkl', 'wb'))

from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pickle file that contains the prediction logic
model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

# Taking User input using forms and passing the values to the predictor
@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('result.html', pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
