import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import pickle
warnings.filterwarnings('ignore')


df = pd.read_csv(r"D:\Projects\heart attack prediction\heart.csv")


X=df.drop("target", axis=1)
y=df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# scaler=StandardScaler()
# scaler.fit(X_train)
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)

sm=SMOTE(random_state=13)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)
#print(np.array(X_test))
RF = RandomForestClassifier(n_estimators=100, random_state=28)
rf_model=RF.fit(X_train_res, y_train_res)
# y_predRF = rf_model.predict_proba(X_test)
#print(y_predRF)
pickle.dump(rf_model,open("modelXG.pkl","wb"))
model=pickle.load(open("modelXG.pkl","rb"))
