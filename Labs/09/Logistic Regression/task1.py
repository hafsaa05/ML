# ===========================
# 1. Import Libraries
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===========================
# 2. Load Dataset
# ===========================
df = pd.read_csv("heart_disease.csv")

# Quick overview
print(df.head())
print(df.info())
print(df.describe())
print("Null values:\n", df.isnull().sum())

# ===========================
# 3. EDA
# ===========================

# Target distribution
sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

# Histograms
df.hist(bins=15, figsize=(15,12), color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Boxplot for outliers
plt.figure(figsize=(12,6))
sns.boxplot(data=df, orient='h')
plt.title("Boxplot of All Features")
plt.show()

# ===========================
# 4. Data Wrangling / Cleaning
# ===========================
# No missing values
# Drop duplicates if any
df.drop_duplicates(inplace=True)

# ===========================
# 5. Feature Scaling
# ===========================
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 6. Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# 7. Logistic Regression with Different Penalties
# ===========================

# ----- L1 Regularization -----
model_l1 = LogisticRegression(penalty='l1', solver='liblinear')  # solver changed to liblinear for L1
model_l1.fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)
acc_train_l1 = model_l1.score(X_train, y_train)
acc_test_l1 = accuracy_score(y_test, y_pred_l1)

# ----- L2 Regularization -----
model_l2 = LogisticRegression(penalty='l2', solver='lbfgs')  # default solver
model_l2.fit(X_train, y_train)
y_pred_l2 = model_l2.predict(X_test)
acc_train_l2 = model_l2.score(X_train, y_train)
acc_test_l2 = accuracy_score(y_test, y_pred_l2)

# ----- ElasticNet Regularization -----
# ElasticNet requires solver='saga' and l1_ratio (mix of L1 and L2)
model_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000)
model_en.fit(X_train, y_train)
y_pred_en = model_en.predict(X_test)
acc_train_en = model_en.score(X_train, y_train)
acc_test_en = accuracy_score(y_test, y_pred_en)

# ===========================
# 8. Compare Results
# ===========================
results = pd.DataFrame({
    "Penalty": ["L1", "L2", "ElasticNet"],
    "Train Accuracy": [acc_train_l1, acc_train_l2, acc_train_en],
    "Test Accuracy": [acc_test_l1, acc_test_l2, acc_test_en]
})

print(results)

# ===========================
# 9. Observations / Comments
# ===========================

"""
Observations:
1. L1 Regularization:
   - Some feature coefficients may become 0 (feature selection)
   - Solver had to be changed to 'liblinear' because 'lbfgs' does not support L1.
2. L2 Regularization:
   - Penalizes large coefficients but keeps all features
   - Default solver 'lbfgs' works fine
3. ElasticNet:
   - Combination of L1 and L2 (controlled by l1_ratio)
   - Requires solver='saga'
   - Needed to increase max_iter to 5000 to ensure convergence (common issue)
   
Errors faced:
- Using L1 or ElasticNet with default solver causes error: 
  "ValueError: 'l1' or 'elasticnet' penalties are not supported by this solver"
- To fix, had to change solver:
    - L1 → solver='liblinear'
    - ElasticNet → solver='saga'
- For ElasticNet, also set l1_ratio to control L1/L2 mix
- max_iter may need increasing if the solver does not converge

Relationship between parameters:
- penalty → Type of regularization
- solver → Optimization algorithm compatible with penalty
- l1_ratio → Only for ElasticNet, balance between L1 and L2
- max_iter → Maximum iterations for convergence
"""
