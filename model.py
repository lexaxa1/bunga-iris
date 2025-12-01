import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


#random seed
seed = 42

#read dataset
iris_df = pd.read_csv("data/iris.csv")
iris_df.sample(frac=1, random_state=seed)

#split dataset into features and target variable
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

#split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

#initialize RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

#train the model
clf.fit(X_train, y_train)

#make predictions
y_pred = clf.predict(X_test)

#evaluate the model 
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

joblib.dump(clf, 'rf_model.sav')