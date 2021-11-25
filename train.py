import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = pd.read_csv("cardio_train.csv", sep=';')
f1scores = []
rocscores = []
recallscores = []
precisionscores = []
data = data.astype(int)
data["cardio"].value_counts()
data["weight_height_index"] = data["weight"]/((data["height"]/100)**2)
one_hot = pd.get_dummies(data['gender'],prefix="gender_")
# Drop column B as it is now encoded
data = data.drop('gender',axis = 1)
# Join the encoded df
data = data.join(one_hot)
one_hot = pd.get_dummies(data['cholesterol'],prefix="cholesterol_")
# Drop column B as it is now encoded
data = data.drop('cholesterol',axis = 1)
# Join the encoded df
data = data.join(one_hot)
one_hot = pd.get_dummies(data['gluc'],prefix="gluc_")
# Drop column B as it is now encoded
data = data.drop('gluc',axis = 1)
# Join the encoded df
data = data.join(one_hot)
data = data[["age", "gender__1","gender__2", "height","weight","weight_height_index","ap_hi","ap_lo","cholesterol__1","cholesterol__2","cholesterol__3","gluc__1","gluc__2","gluc__3","smoke","alco","active","cardio"]]


x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7)
model = xgb.XGBClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
with open("xgboost.dill", "wb") as f:
    dill.dump(model, f)