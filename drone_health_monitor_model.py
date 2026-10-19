# -*- coding: utf-8 -*-
"""
This is test code. We have not released our real data or code. 

The system outputs:

- **Battery Health:** YES / WARNING / NO  
- **Motor Health:** YES / WARNING / NO  
- **Overall System Health:** Score (Out of 100%)

Model: **Random Forest**, because this is a small tabular telemetry dataset.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

"""## 1. Load Dataset

Make sure `drone_health_combined_dataset.csv` is in the same folder as this notebook.
"""

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("drone_health_dataset.csv")

print(df.shape)
df.head()

"""## 2. Check Labels and Basic Dataset Info"""

print(df["label"].value_counts())
print("\nColumns:")
print(df.columns.tolist())

"""## 3. Create Battery, Motor, and Overall Health Labels

Since the current dataset only has `GOOD` and `BAD` labels, we use simple engineering rules to create health categories.

These can be improved later when you collect more detailed fault data.
"""

# Battery Health
battery_health = []

for _, row in df.iterrows():
    voltage = row.get("battery_voltage_mean", 0)

    if row["label"] == "GOOD":
        battery_health.append("YES")
    else:
        # For a 5S battery, around 16.2V or lower is very weak/critical
        if voltage < 16.2:
            battery_health.append("NO")
        else:
            battery_health.append("WARNING")

df["battery_health"] = battery_health


# Motor Health
motor_health = []

for _, row in df.iterrows():
    gyro_mag = row.get("gyro_mag_mean", 0)
    rpm_std = row.get("esc_rpm_std", 0)
    acc_mag = row.get("acc_mag_mean", 0)

    # Simple rule-based motor health estimate
    if gyro_mag < 2 and rpm_std < 100:
        motor_health.append("YES")
    elif gyro_mag < 5 and rpm_std < 500:
        motor_health.append("WARNING")
    else:
        motor_health.append("NO")

df["motor_health"] = motor_health


# Overall Health
overall_health = []

for _, row in df.iterrows():
    battery = row["battery_health"]
    motor = row["motor_health"]

    if battery == "YES" and motor == "YES":
        overall_health.append("HEALTHY")
    elif battery == "NO" or motor == "NO":
        overall_health.append("CRITICAL")
    else:
        overall_health.append("WARNING")

df["overall_health"] = overall_health

df[["label", "battery_health", "motor_health", "overall_health"]].head()

"""## 4. View Health Label Counts"""

print("Battery Health Counts:")
print(df["battery_health"].value_counts())

print("\nMotor Health Counts:")
print(df["motor_health"].value_counts())

print("\nOverall Health Counts:")
print(df["overall_health"].value_counts())

"""## 5. Prepare Features for ML"""

remove_cols = [
    "label",
    "battery_health",
    "motor_health",
    "overall_health"
]

feature_cols = [c for c in df.columns if c not in remove_cols]

X = df[feature_cols].copy()

# Keep ONLY numeric telemetry features
X = X.select_dtypes(include=["number"])

# Fill missing values
X = X.fillna(0)

feature_cols = X.columns.tolist()

battery_target = df["battery_health"]
motor_target = df["motor_health"]
overall_target = df["overall_health"]

print("Number of features:", len(feature_cols))
print(feature_cols)

"""## 6. Train Three Random Forest Models

We train separate models for:

1. Battery health  
2. Motor health  
3. Overall system health
"""

def train_and_evaluate_model(X, y, model_name):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=4,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=" * 60)
    print(model_name)
    print("=" * 60)
    print("Accuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return model, encoder


battery_model, battery_encoder = train_and_evaluate_model(X, battery_target, "Battery Health Model")
motor_model, motor_encoder = train_and_evaluate_model(X, motor_target, "Motor Health Model")
overall_model, overall_encoder = train_and_evaluate_model(X, overall_target, "Overall System Health Model")

"""## 7. Feature Importance for Overall Health"""

importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": overall_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance.head(15)

top_features = importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features for Overall Health Prediction")
plt.tight_layout()
plt.show()

"""## 8. Predict Health for One Sample

This simulates real-time health monitoring.
"""

sample = X.iloc[[0]]

battery_pred = battery_model.predict(sample)
motor_pred = motor_model.predict(sample)
overall_pred = overall_model.predict(sample)

battery_status = battery_encoder.inverse_transform(battery_pred)[0]
motor_status = motor_encoder.inverse_transform(motor_pred)[0]
overall_status = overall_encoder.inverse_transform(overall_pred)[0]

battery_probs = battery_model.predict_proba(sample)[0]
motor_probs = motor_model.predict_proba(sample)[0]

battery_prob_dict = dict(zip(battery_encoder.classes_, battery_probs))
motor_prob_dict = dict(zip(motor_encoder.classes_, motor_probs))

battery_health_score = (
    battery_prob_dict.get("YES", 0) * 100 +
    battery_prob_dict.get("WARNING", 0) * 60 +
    battery_prob_dict.get("NO", 0) * 20
)

motor_health_score = (
    motor_prob_dict.get("YES", 0) * 100 +
    motor_prob_dict.get("WARNING", 0) * 65 +
    motor_prob_dict.get("NO", 0) * 25
)

health_score = round(
    battery_health_score * 0.55 +
    motor_health_score * 0.45,
    1
)

if health_score >= 85:
    final_status = "HEALTHY"
elif health_score >= 60:
    final_status = "WARNING"
else:
    final_status = "CRITICAL"

print("Battery Health:", battery_status)
print("Motor Health:", motor_status)
print("Overall ML Prediction:", overall_status)
print("Battery Health Score:", round(battery_health_score, 1), "/ 100")
print("Motor Health Score:", round(motor_health_score, 1), "/ 100")
print("Overall Health Score:", health_score, "/ 100")
print("Final System Status:", final_status)

"""## 9. Save Predictions for Entire Dataset"""

df["predicted_battery_health"] = battery_encoder.inverse_transform(battery_model.predict(X))
df["predicted_motor_health"] = motor_encoder.inverse_transform(motor_model.predict(X))
df["predicted_overall_health"] = overall_encoder.inverse_transform(overall_model.predict(X))

battery_probs_all = battery_model.predict_proba(X)
motor_probs_all = motor_model.predict_proba(X)

battery_classes = battery_encoder.classes_
motor_classes = motor_encoder.classes_

battery_scores_list = []
motor_scores_list = []
overall_scores_list = []
final_statuses = []

for i in range(len(df)):

    # Battery score
    battery_prob_dict = dict(zip(battery_classes, battery_probs_all[i]))

    battery_health_score = (
        battery_prob_dict.get("YES", 0) * 100 +
        battery_prob_dict.get("WARNING", 0) * 60 +
        battery_prob_dict.get("NO", 0) * 20
    )

    # Motor score
    motor_prob_dict = dict(zip(motor_classes, motor_probs_all[i]))

    motor_health_score = (
        motor_prob_dict.get("YES", 0) * 100 +
        motor_prob_dict.get("WARNING", 0) * 65 +
        motor_prob_dict.get("NO", 0) * 25
    )

    # Overall health score
    health_score = round(
        battery_health_score * 0.55 +
        motor_health_score * 0.45,
        1
    )

    # Final status
    if health_score >= 85:
        final_status = "HEALTHY"
    elif health_score >= 60:
        final_status = "WARNING"
    else:
        final_status = "CRITICAL"

    battery_scores_list.append(round(battery_health_score, 1))
    motor_scores_list.append(round(motor_health_score, 1))
    overall_scores_list.append(health_score)
    final_statuses.append(final_status)

# Save scores into dataframe
df["battery_score"] = battery_scores_list
df["motor_score"] = motor_scores_list
df["health_score"] = overall_scores_list
df["final_system_status"] = final_statuses

df.to_csv("drone_health_predictions.csv", index=False)

df[[
    "label",
    "predicted_battery_health",
    "predicted_motor_health",
    "predicted_overall_health",
    "battery_score",
    "motor_score",
    "health_score",
    "final_system_status"
]].tail(100)

from sklearn.tree import export_text

print("BATTERY MODEL TREES")
for i, tree in enumerate(battery_model.estimators_):
    print(f"\nBATTERY TREE {i}")
    print(export_text(tree, feature_names=feature_cols))

print("MOTOR MODEL TREES")
for i, tree in enumerate(motor_model.estimators_):
    print(f"\nMOTOR TREE {i}")
    print(export_text(tree, feature_names=feature_cols))

print("OVERALL MODEL TREES")
for i, tree in enumerate(overall_model.estimators_):
    print(f"\nOVERALL TREE {i}")
    print(export_text(tree, feature_names=feature_cols))

