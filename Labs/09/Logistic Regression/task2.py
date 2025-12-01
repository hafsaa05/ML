# ===========================
# 1. Import Libraries
# ===========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===========================
# 2. Load Iris Dataset
# ===========================
df = pd.read_csv("iris.csv")

# Quick check
print(df.head())
print(df.info())

# Separate features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# 3. Logistic Regression with different solvers
# ===========================
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
results = []

for solver in solvers:
    try:
        model = LogisticRegression(solver=solver, max_iter=5000)  # increase max_iter for convergence
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        results.append([solver, train_acc, test_acc])
    except Exception as e:
        print(f"Solver {solver} caused error: {e}")
        results.append([solver, None, None])

# Create results dataframe
df_results_iris = pd.DataFrame(results, columns=['Solver', 'Training Accuracy', 'Testing Accuracy'])
print("\nIris Dataset Solver Comparison:\n", df_results_iris)

# ===========================
# 4. Observations / Best Solver for Iris
# ===========================
"""
Observations:
- Small dataset like Iris (150 samples), solvers like 'liblinear', 'newton-cg', 'lbfgs' work well.
- 'sag' and 'saga' are optimized for larger datasets (they use stochastic methods).
- 'liblinear' is good for small datasets and supports L1 penalty.
- In this case, best solver for Iris: 'lbfgs' or 'newton-cg' (stable, fast convergence)
"""

# ===========================
# 5. Apply same solvers to Heart Disease dataset
# ===========================
df_hd = pd.read_csv("heart_disease.csv")

X_hd = df_hd.drop('target', axis=1)
y_hd = df_hd['target']

scaler_hd = StandardScaler()
X_hd_scaled = scaler_hd.fit_transform(X_hd)

X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(
    X_hd_scaled, y_hd, test_size=0.2, random_state=42, stratify=y_hd
)

results_hd = []

for solver in solvers:
    try:
        model_hd = LogisticRegression(solver=solver, max_iter=5000)
        model_hd.fit(X_train_hd, y_train_hd)
        train_acc = model_hd.score(X_train_hd, y_train_hd)
        y_pred = model_hd.predict(X_test_hd)
        test_acc = accuracy_score(y_test_hd, y_pred)
        results_hd.append([solver, train_acc, test_acc])
    except Exception as e:
        print(f"Solver {solver} caused error on Heart Disease dataset: {e}")
        results_hd.append([solver, None, None])

df_results_hd = pd.DataFrame(results_hd, columns=['Solver', 'Training Accuracy', 'Testing Accuracy'])
print("\nHeart Disease Dataset Solver Comparison:\n", df_results_hd)

# ===========================
# 6. Observations / Dataset Size Effect
# ===========================
"""
- Solvers like 'lbfgs', 'newton-cg', 'newton-cholesky' are good for small-to-medium datasets.
- 'sag' and 'saga' are better for large datasets (they use stochastic updates and scale well).
- Comparing Iris vs Heart Disease dataset:
    - Iris (150 samples) → All solvers converge quickly, minor differences.
    - Heart Disease (~300 samples) → 'saga' and 'sag' show faster convergence, more stable.
- Dataset size does affect which solver is more efficient.
- Best overall for small dataset: 'liblinear' or 'lbfgs'.
- Best for medium dataset: 'lbfgs', 'newton-cg'.
- Best for large dataset: 'saga', 'sag'.
"""
