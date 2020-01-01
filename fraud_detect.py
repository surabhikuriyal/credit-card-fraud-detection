import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

df = pd.read_csv('creditcard.csv')
class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))

feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:1, 30: ].columns

data_features = df[feature_names]
data_target = df[target]

from sklearn.model_selection import train_test_split
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(data_features,    data_target, train_size=0.70, test_size=0.30, random_state=1)

from sklearn.ensemble import RandomForestClassifier

#training the model and generating predictions :
model = RandomForestClassifier(n_estimators = 100, n_jobs =4)
model.fit(X_train, y_train.values.ravel())
pred = model.predict(X_test)
confmat = confusion_matrix(y_test, pred)
import pickle
with open('my_model','wb')as f:
    pickle.dump(model,f)

# separate out the confusion matrix components :
tpos = confmat[0][0]
fneg = confmat[1][1]
fpos = confmat[0][1]
tneg = confmat[1][0]

recallScore = round(recall_score(y_test, pred), 2)
# calculate and display metrics
print(cmat)
print( 'Accuracy of the model : '+ str(np.round(100*float(tpos+fneg)/float(tpos+fneg + fpos + tneg),2))+'%')
print("Sensitivity/Recall for Model/correctly predicted positive observations to the all observations : {recall_score}".format(recall_score = recallScore))
print("\n---MODEL CREATED---")

