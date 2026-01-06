import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

print("=== START TRAIN SCRIPT ===")

# 1. Load data yang sudah clean + target
df = pd.read_csv("train_cleaned_with_target.csv")

target_col = "status_group"

# Kolom yang ingin dibuang (ID & yang redundant)
drop_cols = ["id", "date_recorded", "year_recorded", "construction_year"]

y = df[target_col]
X = df.drop(columns=drop_cols + [target_col])

# 2. Pisah numerical & categorical
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("Numerical cols :", num_cols)
print("Categorical cols:", cat_cols)

# 3. Train–valid split (supaya kita bisa cek performa sedikit)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough"
)

# 5. Model Random Forest (versi ringan)
rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# 6. Train model
print("Training model...")
rf_model.fit(X_train, y_train)

# 7. Evaluasi cepat di validation set
y_pred = rf_model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
f1_macro = f1_score(y_valid, y_pred, average="macro")

print("\n=== Validation Performance (Quick RF) ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Macro F1  : {f1_macro:.4f}\n")
print(classification_report(y_valid, y_pred))

# 8. Fit ulang ke seluruh data (optional tapi bagus untuk final model)
print("Refit model ke seluruh data...")
rf_model.fit(X, y)

# 9. Simpan model ke .pkl (INI YANG DIPAKAI STREAMLIT)
joblib.dump(rf_model, "waterpoint_streamlit_model.pkl")
print("Model saved to waterpoint_streamlit_model.pkl")

print("=== END TRAIN SCRIPT ===")