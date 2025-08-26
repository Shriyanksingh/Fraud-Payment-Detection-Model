# Fraud Detection Model using Gradient Boosting(XGBoost)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier

# Load the data
df = pd.read_csv("Fraud.csv")
print("Dataset Shape:", df.shape)


# Data Cleaning
print("Missing values:\n", df.isnull().sum())

# Drop indentifeirs column
df = df.drop(["nameOrig", "nameDest"], axis=1)

# Encode categorical 'type'
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Correlation check(multi-collinearity)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Feature correlation heatmap")
plt.show()

#Feature & Target
X = df.drop(["isFraud", "isFlaggedFraud"], axis=1)  
y = df["isFraud"]                                  

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Gradient Boosting Model (XGBoost)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

xgb.fit(X_train, y_train)

# Model Evaluation
y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)[:,1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve (AUC=%.4f)" % pr_auc)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)

sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance from XGBoost")
plt.show()





'''Answers to the given Question

