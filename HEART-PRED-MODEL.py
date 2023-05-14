import pandas as pd
import numpy as np

# import dataset
df = pd.read_csv("/Users/ohenmaao/Desktop/DATASETS/processed_cleveland.csv")

# clean dataset
df = df.replace('?', np.nan)
df = df.dropna()

# split into features and labels
X = df.iloc[:, :-1]
y = df['num']

# convert categorical values to numerical variables
X = pd.get_dummies(X, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])

# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

#tree
tree = DecisionTreeClassifier(random_state=42)

# train model
tree.fit(X_train, y_train)


# new patient's data prediction
new_patient_data = [[58, 1, 0, 150, 283, 1, 0, 162, 0, 1.0, 1, 0, 3]]

# convert categorical values to numerical variables
new_patient_data = pd.DataFrame(new_patient_data, columns=df.columns[:-1])
new_patient_data = pd.get_dummies(new_patient_data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])

# align columns
new_patient_data = new_patient_data.reindex(columns=X.columns, fill_value=0)

# normalize data
new_patient_data = scaler.transform(new_patient_data)

# make prediction
prediction = tree.predict(new_patient_data)

# print prediction
if prediction == 0:
    print("The patient does not have heart disease.")
else:
    print("The patient has heart disease.")

# testing accuracy
accuracy = tree.score(X_test, y_test)
print("Accuracy on test data: {:.2f}%".format(accuracy * 100))

# only 55% accurate, can be better, did get some predicts wrong
# will have to come back and maybe try a random forrest classifier instead


