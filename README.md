# Fraudulent Transaction Detection using Gradient Boosting

## 📌 Project Overview
This project develops a machine learning solution to proactively **detect fraudulent financial transactions**.  
Using a dataset of over **6 million transactions and 10 features**, the project demonstrates how to build a robust fraud detection pipeline with **XGBoost (Gradient Boosting)**.

The pipeline covers:
- Data cleaning (missing values, outliers, multicollinearity)
- Feature engineering (encoding transaction type, creating balance-difference features)
- Model training with XGBoost, tuned for imbalanced datasets
- Model evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Extraction of feature importance and business insights
- Recommendations for fraud prevention strategies

---

## 📂 Dataset
- **Transactions:** ~6.3 million rows, 10 columns  
- **Key features:**
  - `step`: Hourly time step in the dataset
  - `type`: Transaction type (CASH-IN, CASH-OUT, TRANSFER, etc.)
  - `amount`: Transaction amount
  - `oldbalanceOrg`, `newbalanceOrg`: Sender’s balance before/after
  - `oldbalanceDest`, `newbalanceDest`: Recipient’s balance before/after
  - `isFraud`: Target variable (1 = Fraud, 0 = Legit)
  - `isFlaggedFraud`: Flag for illegal attempts (>200k transfers)

Data dictionary reference is included in **Data Dictionary.txt**.

---

## 🚀 Model Development
1. **Data Cleaning**
   - Checked and handled missing values
   - Analyzed outliers
   - Checked for multicollinearity(correlation heatmap, VIF)
2. **Feature Engineering**
   - Dropped IDs (`nameOrig`, `nameDest`)
   - Encoded categorical variable `type`
   - Created engineered features: `deltaOrg`, `deltaDest`
3. **Model Training**
   - Train/test split (80/20)
   - Gradient Boosting using XGBoost
   - Addressed class imbalance with `scale_pos_weight`
4. **Model Evaluation**
   - Classification Report(Precision, Recall, F1)
   - Confusion Matrix
   - ROC-AUC and Precision-Recall AUC
5. **Insights**
   - Transaction type, amount, and balance inconsistencies are strong fraud predictors

---

## 📊 Results
- **High ROC-AUC (>0.95) and PR-AUC**, showing strong ability to distinguish fraud vs. legitimate transactions.
- Fraud is concentrated in **TRANSFER** and **CASH-OUT** transactions.
- Large transaction amounts and balance mismatches strongly indicate fraud.

---

## 🛡️ Business Recommendations
- Real-time monitoring of high-value transfers
- Strong authentication for risky transactions (OTP, biometrics)
- Balance anomaly detection and velocity rules
- Continuous monitoring of fraud rate trends after prevention measures

---

## 📁 Repository Structure
```
├── Task Details.pdf          # Instructions and task description
├── Data Dictionary.txt       # Data dictionary of dataset features
├── sher1.xlsx                # Transaction dataset (6M+ rows)
├── Fraud_Detection_GradientBoosting.ipynb  # Full Jupyter Notebook with code & answers
├── README.md                 # Project overview and instructions
```

---

## ⚙️ Requirements
- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn, xgboost, statsmodels (optional)

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 How to Run
1. Open the Jupyter Notebook:
```bash
jupyter notebook Fraud_Detection_GradientBoosting.ipynb
```
2. Run all cells sequentially.
3. View evaluation metrics and feature importance charts.

---

## 📌 Author
- **Shriyank Singh**  
- Created as part of a fraud detection case study project.
