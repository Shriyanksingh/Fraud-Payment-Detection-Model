#Fraud Detection Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import Train_Test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,precision_recall_curve,auc
from xgboost import XGBClassifier

#Data loading and cleaning
df= pd.read_csv("Dataset.csv")
print("Dataset Shape:", df.shape)
print("Missing values:\n",df.isnull().sum())
df= df.drop(["nameOrig", "nameDest"], axis=1)
le= LabelEncoder()
df['type']= le.fit_transform(df['type'])
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature correlatian heatmap")
plt.show()
X= df.drop(["isFraud", "isFlaggedFraud"], axis=1)  
y= df["isFraud"]                                  

#Train-Test Split
X_Train, X_Test, y_Train, y_Test = Train_Test_split(X, y, Test_size=0.2, random_state=42, stratify=y)

#Gradient Boosting Model(XGBoost)
Scale_pos_weight= (len(y_Train) - sum(y_Train))/sum(y_Train)
xgb= XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,
    Scale_pos_weight=Scale_pos_weight,eval_metric="logloss",random_state=42,use_label_encoder=False
)
xgb.fit(X_Train, y_Train)

#Model evaluation and Confusion matrix
y_pred= xgb.predict(X_Test)
y_pred_proba= xgb.predict_proba(X_Test)[:,1]
print("\nClassification Report:\n", classification_report(y_Test,y_pred))
print("ROC-AUC :",roc_auc_score(y_Test, y_pred_proba))
cm= confusion_matrix(y_Test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Precision-Recall Curve
precision,recall,thresholds= precision_recall_curve(y_Test, y_pred_proba)
pr_auc= auc(recall, precision)
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve (AUC=%.4f)" % pr_auc)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Feature importance
importance= pd.DataFrame({"Feature": X.columns,"importance": xgb.feature_importances_}).sort_values(by="importance", ascending=False)
print("\nFeature importance:\n", importance)
sns.barplot(x= "importance",y="feature", data= importance)
plt.title("Feature importance from XGBoost")
plt.show()








