import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset1
df1 = pd.read_csv("students_.csv")
X1 = df1.drop("Target", axis=1)
y1 = LabelEncoder().fit_transform(df1["Target"])
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Dataset2
df2 = pd.read_csv("healthy aging.csv")
X2 = df2.drop("Trouble Sleeping", axis=1)
y2 = df2["Trouble Sleeping"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

Dataset1: Decision Trees

# ----- Default Gini -----
dt_default = DecisionTreeClassifier()
dt_default.fit(X1_train, y1_train)
pred_default = dt_default.predict(X1_test)
print("\n========= Dataset1: Decision Tree (Default Gini) =========")
print("Training Accuracy:", dt_default.score(X1_train, y1_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y1_test, pred_default) * 100, "%")

# ----- Gini with Pruning -----
dt_gini_pruned = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.015)
dt_gini_pruned.fit(X1_train, y1_train)
pred_gini_pruned = dt_gini_pruned.predict(X1_test)
print("\n========= Dataset1: Decision Tree (Gini + Pruning) =========")
print("Training Accuracy:", dt_gini_pruned.score(X1_train, y1_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y1_test, pred_gini_pruned) * 100, "%")

# ----- Entropy -----
dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X1_train, y1_train)
pred_entropy = dt_entropy.predict(X1_test)
print("\n========= Dataset1: Decision Tree (Entropy) =========")
print("Training Accuracy:", dt_entropy.score(X1_train, y1_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y1_test, pred_entropy) * 100, "%")

# ----- Entropy with Pruning -----
dt_entropy_pruned1 = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.015)
dt_entropy_pruned1.fit(X1_train, y1_train)
pred_entropy_pruned1 = dt_entropy_pruned1.predict(X1_test)
print("\n========= Dataset1: Decision Tree (Entropy + Pruning) =========")
print("Training Accuracy:", dt_entropy_pruned1.score(X1_train, y1_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y1_test, pred_entropy_pruned1) * 100, "%")

# Dataset2: Decision Trees

# ----- Default Gini -----
dt_default2 = DecisionTreeClassifier()
dt_default2.fit(X2_train, y2_train)
pred_default2 = dt_default2.predict(X2_test)
print("\n========= Dataset2: Decision Tree (Default Gini) =========")
print("Training Accuracy:", dt_default2.score(X2_train, y2_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y2_test, pred_default2) * 100, "%")

# ----- Gini with Pruning -----
dt_gini_pruned2 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.015)
dt_gini_pruned2.fit(X2_train, y2_train)
pred_gini_pruned2 = dt_gini_pruned2.predict(X2_test)
print("\n========= Dataset2: Decision Tree (Gini + Pruning) =========")
print("Training Accuracy:", dt_gini_pruned2.score(X2_train, y2_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y2_test, pred_gini_pruned2) * 100, "%")

# ----- Entropy -----
dt_entropy2 = DecisionTreeClassifier(criterion='entropy')
dt_entropy2.fit(X2_train, y2_train)
pred_entropy2 = dt_entropy2.predict(X2_test)
print("\n========= Dataset2: Decision Tree (Entropy) =========")
print("Training Accuracy:", dt_entropy2.score(X2_train, y2_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y2_test, pred_entropy2) * 100, "%")

# ----- Entropy with Pruning -----
dt_entropy_pruned2 = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.015)
dt_entropy_pruned2.fit(X2_train, y2_train)
pred_entropy_pruned2 = dt_entropy_pruned2.predict(X2_test)
print("\n========= Dataset2: Decision Tree (Entropy + Pruning) =========")
print("Training Accuracy:", dt_entropy_pruned2.score(X2_train, y2_train) * 100, "%")
print("Testing Accuracy :", accuracy_score(y2_test, pred_entropy_pruned2) * 100, "%")

#  Actual vs Predicted (Dataset2, Entropy + Pruning)

comparison_df = pd.DataFrame({
    'Actual': y2_test,
    'Predicted': pred_entropy_pruned2
})

correct = (comparison_df['Actual'] == comparison_df['Predicted']).sum()
wrong = len(comparison_df) - correct

print(f"\nCorrectly Predict: {correct} Out of {len(comparison_df)}")
print(f"Wrong Predictions: {wrong}")

# Visualization (Entropy + Pruning, Dataset2)
plt.figure(figsize=(12, 8))
plot_tree(
    dt_entropy_pruned2,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Dataset2: Decision Tree Visualization (Entropy + Pruning)")
plt.show()

#  Accuracy Bar Plots
dataset1_models = [("Default Gini", dt_default),
                   ("Gini Pruned", dt_gini_pruned),
                   ("Entropy", dt_entropy),
                   ("Entropy Pruned", dt_entropy_pruned1)]

dataset2_models = [("Default Gini", dt_default2),
                   ("Gini Pruned", dt_gini_pruned2),
                   ("Entropy", dt_entropy2),
                   ("Entropy Pruned", dt_entropy_pruned2)]

# Plot Dataset1
plt.figure(figsize=(10,5))
for name, model in dataset1_models:
    plt.bar(name + " Train", model.score(X1_train, y1_train), color='skyblue')
    plt.bar(name + " Test", model.score(X1_test, y1_test), color='salmon')
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Dataset1: Training vs Testing Accuracy")
plt.xticks(rotation=45)
plt.show()

# Plot Dataset2
plt.figure(figsize=(10,5))
for name, model in dataset2_models:
    plt.bar(name + " Train", model.score(X2_train, y2_train), color='skyblue')
    plt.bar(name + " Test", model.score(X2_test, y2_test), color='salmon')
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Dataset2: Training vs Testing Accuracy")
plt.xticks(rotation=45)
plt.show()
