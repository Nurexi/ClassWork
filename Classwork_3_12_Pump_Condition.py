# Classwork 3.12 - Pump condition classification (Normal / Warning / Fault)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

classes = ["Normal", "Warning", "Fault"]

# 1. dataset (45 pump records)

data = [
    # ID, Vibration_mm_s, Speed_RPM, Temperature_C, Pressure_bar, Operating_Hours, Condition
    [1,  1.2, 1490, 50, 5.2,  800, "Normal"],
    [2,  1.5, 1485, 53, 5.1, 1200, "Normal"],
    [3,  1.8, 1500, 56, 5.3, 1600, "Normal"],
    [4,  2.0, 1475, 58, 5.0, 1900, "Normal"],
    [5,  2.2, 1495, 60, 5.2, 2100, "Normal"],
    [6,  1.4, 1505, 52, 5.4, 1000, "Normal"],
    [7,  2.5, 1460, 62, 4.9, 2400, "Normal"],
    [8,  1.7, 1498, 55, 5.3, 1450, "Normal"],
    [9,  2.8, 1445, 65, 4.9, 2700, "Normal"],
    [10, 1.3, 1502, 49, 5.5,  700, "Normal"],
    [11, 2.1, 1480, 59, 5.1, 2000, "Normal"],
    [12, 2.4, 1465, 61, 5.0, 2300, "Normal"],
    [13, 1.9, 1492, 57, 5.2, 1750, "Normal"],
    [14, 3.0, 1440, 66, 4.8, 2900, "Normal"],
    [15, 3.2, 1435, 68, 4.8, 3100, "Normal"],
    [16, 3.6, 1450, 70, 4.7, 3300, "Warning"],
    [17, 4.0, 1425, 72, 4.6, 3800, "Warning"],
    [18, 4.4, 1415, 74, 4.5, 4200, "Warning"],
    [19, 4.8, 1405, 76, 4.4, 4600, "Warning"],
    [20, 5.1, 1395, 78, 4.4, 5000, "Warning"],
    [21, 3.9, 1430, 71, 4.7, 3600, "Warning"],
    [22, 5.4, 1385, 80, 4.3, 5300, "Warning"],
    [23, 4.2, 1420, 73, 4.6, 4000, "Warning"],
    [24, 5.7, 1375, 82, 4.2, 5600, "Warning"],
    [25, 4.6, 1410, 75, 4.5, 4400, "Warning"],
    [26, 5.0, 1400, 77, 4.4, 4800, "Warning"],
    [27, 3.8, 1435, 69, 4.6, 3500, "Warning"],
    [28, 5.9, 1370, 84, 4.2, 5900, "Warning"],
    [29, 4.5, 1412, 76, 4.5, 4300, "Warning"],
    [30, 5.3, 1390, 79, 4.3, 5100, "Warning"],
    [31, 6.0, 1365, 82, 4.2, 5400, "Fault"],
    [32, 6.5, 1345, 86, 4.0, 6000, "Fault"],
    [33, 7.1, 1320, 90, 3.8, 6800, "Fault"],
    [34, 7.8, 1290, 95, 3.6, 7500, "Fault"],
    [35, 6.2, 1355, 84, 4.1, 5700, "Fault"],
    [36, 8.4, 1260, 99, 3.4, 8100, "Fault"],
    [37, 6.8, 1330, 88, 3.9, 6400, "Fault"],
    [38, 8.8, 1240, 102, 3.3, 8600, "Fault"],
    [39, 7.4, 1305, 92, 3.7, 7100, "Fault"],
    [40, 6.6, 1340, 87, 4.0, 6200, "Fault"],
    [41, 8.1, 1270, 97, 3.5, 7900, "Fault"],
    [42, 9.2, 1225, 104, 3.2, 8900, "Fault"],
    [43, 7.0, 1315, 89, 3.8, 6600, "Fault"],
    [44, 5.8, 1372, 81, 4.3, 5300, "Fault"],
    [45, 7.6, 1295, 94, 3.6, 7300, "Fault"],
]

columns = ["Pump_ID", "Vibration_mm_s", "Speed_RPM", "Temperature_C",
           "Pressure_bar", "Operating_Hours", "Condition"]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET")
print(df.to_string(index=False))
print("\nMissing values:")
print(df.isna().sum())
print("\nClass distribution:")
print(df["Condition"].value_counts().reindex(classes))

feature_cols = ["Vibration_mm_s", "Speed_RPM", "Temperature_C",
                "Pressure_bar", "Operating_Hours"]
X = df[feature_cols]
y = df["Condition"]

# plots of the data
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Temperature_C", y="Vibration_mm_s",
                hue="Condition", hue_order=classes,
                style="Condition", style_order=classes, s=100)
plt.title("Temperature vs vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.tight_layout()
plt.show()

corr_df = df[feature_cols].copy()
corr_df["Condition"] = df["Condition"].map({"Normal": 0, "Warning": 1, "Fault": 2})
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

# 2. train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# 3. models

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "AdaBoost": adaboost_model
}

# 4. train and evaluate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for model_name, model in models.items():
    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    fault_recall = recall_score(y_test, y_pred, labels=["Fault"], average=None, zero_division=0)[0]
    fault_precision = precision_score(y_test, y_pred, labels=["Fault"], average=None, zero_division=0)[0]
    macro_f1 = f1_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro").mean()

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Fault Recall": fault_recall,
        "Fault Precision": fault_precision,
        "Macro F1": macro_f1,
        "CV Macro F1": cv_f1
    })

    print(f"Accuracy         : {accuracy:.3f}")
    print(f"Recall (Fault)   : {fault_recall:.3f}")
    print(f"Precision (Fault): {fault_precision:.3f}")
    print(f"Macro F1         : {macro_f1:.3f}")
    print(f"CV Macro F1      : {cv_f1:.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=classes,
                                target_names=classes, zero_division=0))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=classes),
        index=["Actual " + c for c in classes],
        columns=["Pred " + c for c in classes]
    ))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=classes, display_labels=classes,
        cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()

# 5. compare the models

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON")
print(results_df.round(3).to_string())

results_df[["Fault Recall", "Fault Precision", "Macro F1"]].plot(
    kind="bar", figsize=(11, 6), ylim=(0, 1.05)
)
plt.title("Model comparison")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 6. feature importance

for name, model in {"Decision Tree": decision_tree_model,
                    "Random Forest": random_forest_model}.items():
    importance = pd.Series(model.feature_importances_,
                           index=feature_cols).sort_values(ascending=False)
    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index[::-1], importance.values[::-1])
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(16, 8))
plot_tree(decision_tree_model, feature_names=feature_cols,
          class_names=decision_tree_model.classes_,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

# 7. new pumps

new_pumps = pd.DataFrame([
    {"Vibration_mm_s": 1.6, "Speed_RPM": 1492, "Temperature_C": 54, "Pressure_bar": 5.2, "Operating_Hours": 1300},
    {"Vibration_mm_s": 6.2, "Speed_RPM": 1355, "Temperature_C": 79, "Pressure_bar": 4.2, "Operating_Hours": 5800},
    {"Vibration_mm_s": 8.0, "Speed_RPM": 1275, "Temperature_C": 96, "Pressure_bar": 3.5, "Operating_Hours": 7800},
], index=["Pump A", "Pump B", "Pump C"])

print("\nNEW PUMPS")
print(new_pumps)

predicted = pd.DataFrame(
    {name: model.predict(new_pumps) for name, model in models.items()},
    index=new_pumps.index
)
print("\nPredicted condition:")
print(predicted)
