Step 1
# Import important labaries and Read dataset
import pandas as pd
import numpy as np

# read dataset
data = pd.read_csv(r"C:\Users\bsrav\OneDrive\Desktop\PS_20174392719_1491204439457_log.csv")
print(data.head())

step 2
# check dataset has any null value or not
print (data.isnull().sum())

step 3
# Exploring transaction type
print (data.type.value_counts())

step 4
type = data["type"].value_counts()
transactions = type.index
quantity = type.values
import plotly.express as px
figure = px.pie(data, values=quantity ,names=transactions,hole = 0.5,title="Distribution of Transcation Type")
figure.show()

step 5
# check correlation b/w the feature of the data with the fraud column

# checking correlation
correlation = data.corr()
print (correlation["isFraud"].sort_values(ascending=False))

step 6
data["type"]= data["type"].map({"CASH_OUT": 1,"PAYMENT": 2,"CASH_IN": 3,"TRANSFER": 4,"DEBIT": 5})
data["isFraud"]=data["isFraud"].map({0:"No Fraud",1:"Fraud"})
print (data.head())

step 7
# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]])
y = np.array(data[["isFraud"]])

step 8
# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split (x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

step 9
# prediction
# feature=[type,amount,oldbalanceOrg,newbalanceOrig]
features= np.array([[4,9000.60,9000.60,0.0]])
print(model.predict(features))