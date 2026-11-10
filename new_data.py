"""
Practical Engineering Case Study: Supervised Machine Learning Classification
Algorithms:
1. Support Vector Machines (SVM)
2. Naïve Bayes (Gaussian Naïve Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)  <-- NEW DATA
# ============================================================

data = [
    # ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk
    [1,  46, 1.1, 1495,  700, 0],
    [2,  52, 1.6, 1485, 1300, 0],
    [3,  57, 1.9, 1470, 1700, 0],
    [4,  59, 2.1, 1505, 2000, 0],
    [5,  63, 2.5, 1455, 2400, 0],
    [6,  54, 1.5, 1500, 1050, 0],
    [7,  65, 2.7, 1445, 2600, 0],
    [8,  49, 1.3, 1492,  850, 0],
    [9,  61, 2.2, 1478, 2150, 0],
    [10, 55, 1.7, 1488, 1450, 0],
    [11, 68, 3.0, 1430, 2900, 0],
    [12, 58, 2.0, 1495, 1850, 0],
    [13, 64, 2.4, 1460, 2350, 0],
    [14, 51, 1.4, 1502, 1150, 0],
    [15, 67, 3.1, 1425, 3300, 0],
    [16, 80, 5.4, 1385, 4900, 1],
    [17, 86, 6.3, 1350, 5800, 1],
    [18, 92, 7.3, 1315, 6900, 1],
    [19, 74, 4.7, 1410, 4300, 1],
    [20, 96, 7.9, 1275, 7500, 1],
    [21, 83, 5.9, 1365, 5500, 1],
    [22, 100, 8.6, 1250, 8300, 1],
    [23, 72, 4.4, 1415, 3800, 1],
    [24, 89, 6.9, 1320, 6700, 1],
    [25, 85, 6.5, 1340, 6200, 1],
    [26, 104, 9.0, 1235, 8800, 1],
    [27, 78, 5.0, 1395, 4600, 1],
    [28, 94, 7.6, 1285, 7700, 1],
    [29, 82, 5.7, 1360, 5600, 1],
    [30, 98, 8.3, 1265, 8000, 1],
]

columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "Speed_RPM", "Operating_Hours", "Failure_Risk"
]
df = pd.DataFrame(data, columns=columns)

print("\\nDATASET: 30 MACHINE RECORDS")
print(df.to_string(index=False))
print("\\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "Failure risk"}))

# Separate predictors (X) and target (y).
feature_cols = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"]
X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", style="Failure_Risk", s=100
)
plt.title("Machine temperature vs. vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of engineering variables")
plt.tight_layout()
plt.show()

# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

# Stratification preserves the class ratio in both subsets.
# With only 30 records, test metrics will be highly sensitive to individual records.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

# SVM: scaling is important because SVM uses distances/margins.
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Gaussian Naive Bayes: assumes feature values are conditionally Gaussian
# and conditionally independent given the class.
naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

# Decision Tree: learns a sequence of feature-based splitting rules.
decision_tree_model = DecisionTreeClassifier(
    max_depth=3, random_state=42
)

# Random Forest: combines many decision trees trained on bootstrap samples.
random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

# AdaBoost: sequentially trains weak learners, emphasizing previously
# misclassified training examples.
# SAMME is explicitly selected for compatibility across sklearn versions.
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():
    print("\\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    # Learn from the training records.
    model.fit(X_train, y_train)

    # Predict labels for unseen test records.
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure risk"],
        zero_division=0
    ))

    # Confusion matrix visualization for this algorithm.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Failure risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of five supervised learning algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Engineering feature")
    plt.tight_layout()
    plt.show()

# Visualize the learned Decision Tree.
plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree learned from the training records")
plt.tight_layout()
plt.show()

# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================

# Example values are illustrative. They are not a real machine diagnosis only for you(Software Engineering 4th year students).
new_machine = pd.DataFrame([{
    "Temperature_C": 89,
    "Vibration_mm_s": 6.8,
    "Speed_RPM": 1325,
    "Operating_Hours": 6500
}])

print("\\nNEW MACHINE OBSERVATION")
print(new_machine)

for name, model in models.items():
    predicted_class = int(model.predict(new_machine)[0])
    label = "Failure risk" if predicted_class == 1 else "Normal"
    print(f"{name:15s} prediction: {label}")

print("\\nEND OF PRACTICAL CASE STUDY")
