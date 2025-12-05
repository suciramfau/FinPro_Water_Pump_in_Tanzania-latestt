import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train_cleaned_with_target.csv")

target_col = "status_group"
drop_cols = ["id", "date_recorded", "year_recorded", "construction_year"]

X = df.drop(columns=[target_col] + drop_cols)
y = df[target_col]

num_cols = ["amount_tsh","gps_height","longitude","latitude","population","pump_age"]
cat_cols = [
    "basin","region_code","district_code","public_meeting","permit",
    "extraction_type_class","management","payment_type","water_quality",
    "quantity","source_type","source_class","waterpoint_type"
]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

rf_small = RandomForestClassifier(
    n_estimators=100,   # lebih kecil
    max_depth=20,       # dibatasi
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

from sklearn.metrics import accuracy_score, f1_score

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", rf_small)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("Macro F1:", f1_score(y_valid, y_pred, average="macro"))

joblib.dump(pipe, "waterpoint_small_model.pkl")
print("Saved to waterpoint_small_model.pkl")
