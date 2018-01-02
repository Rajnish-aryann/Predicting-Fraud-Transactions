import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
os.chdir("/home/debadri/Downloads/BW_Class")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
total=train.append(test)

total["cat_var_1"]=total["cat_var_1"].fillna("-1")
total["cat_var_3"]=total["cat_var_3"].fillna("-1")
total["cat_var_6"]=total["cat_var_6"].fillna("-1")
total["cat_var_8"]=total["cat_var_8"].fillna("-1")

#variables with only one class
del total["cat_var_31"]
del total["cat_var_35"]
del total["cat_var_36"]
del total["cat_var_37"]
del total["cat_var_38"]
del total["cat_var_40"]
del total["cat_var_42"]

#variables woth one class having very less number of instances
del total["cat_var_41"]
del total["cat_var_39"]
del total["cat_var_34"]
del total["cat_var_33"]
del total["cat_var_32"]
del total["cat_var_30"]
del total["cat_var_29"]
del total["cat_var_28"]
del total["cat_var_27"]
# var 26 and 25 also have very less insctances od 1 woth 90% chance of target 1

#feature with extreme low importances


##

#del total["cat_var_20"]
#del total["cat_var_23"]
#del total["cat_var_4"]

del total["cat_var_22"]

del total["num_var_3"]

del total["transaction_id"]
total["num_var_1"]=total["num_var_1"].transform(lambda x:np.log(x))
total["num_var_7"]=total["num_var_7"].transform(lambda x:np.log(x))
total["num_var_4"]=total["num_var_4"].transform(lambda x:np.log(x))
total["num_var_2"]=total["num_var_2"].transform(lambda x:10**x)



lbl=LabelEncoder()
col=total.columns
for i in col:
    if train[i].dtype=='object':
        total[i]=total[i].astype(str)
        total[i]=lbl.fit_transform(total[i])
        
y=total["target"]
Y=y[:348978]
Y=Y.astype(int)
del total["target"]
X_train=total.iloc[:348978,:]
X_test=total.iloc[348978:,:]
clf1 = xgb.XGBClassifier(max_depth=7,n_estimators=350,learning_rate=0.05,scale_pos_weight=6).fit(X_train,Y)        
p=clf1.predict_proba(X_test)
pred=[row[1] for row in p] 
sample=pd.read_csv("sample_submissions.csv")
sample["target"]=pred

sample.to_csv("submit2.csv",index=False)


xgb.plot_importance(clf1)


xgb.plot_importance(clf1)
