import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# CONFIG
# ================================
st.set_page_config(
    page_title="Tanzania Water Pump App",
    layout="wide"
)

# ================================
# CONSTANTS (MODEL FEATURES)
# (based on training pipeline)
# ================================
NUM_COLS = ["amount_tsh", "gps_height", "longitude", "latitude", "population", "pump_age"]
CAT_COLS = [
    "basin", "region_code", "district_code", "public_meeting", "permit",
    "extraction_type_class", "management", "payment_type", "water_quality",
    "quantity", "source_type", "source_class", "waterpoint_type"
]
REQUIRED_FEATURES = NUM_COLS + CAT_COLS

# ================================
# CACHE: LOAD MODEL & DATA
# ================================
@st.cache_resource
def load_model():
    model = joblib.load("waterpoint_small_model.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("train_cleaned_with_target.csv")
    return df

# ================================
# HELPERS
# ================================
def make_dummy_csv_bytes(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    df = df[REQUIRED_FEATURES]  
    return df.to_csv(index=False).encode("utf-8")

# ================================
# HELPERS
# ================================
def make_dummy_csv_bytes(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    df = df[REQUIRED_FEATURES]  
    return df.to_csv(index=False).encode("utf-8")


def validate_and_prepare_batch_df(batch_df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Validates columns, drops extras, and returns a prepared DF
    that matches the model's expected input schema.
    Always returns: (ok, msg, prepared_df_or_none)
    """
    # Guard: None / not a DataFrame
    if batch_df is None:
        return False, "There is no data. Please upload a CSV file first", None

    # Guard: empty df
    if batch_df.empty:
        return False, "CSV kosong. Silakan upload file yang berisi data.", None

    cols = list(batch_df.columns)

    missing = [c for c in REQUIRED_FEATURES if c not in cols]
    if missing:
        msg = (
            "The CSV file is empty. Please upload a file that contains data.\n\n"
            f"Required columns are missing({len(missing)}):\n- "
            + "\n- ".join(missing)
        )
        return False, msg, None

    # Drop target/metadata columns if user accidentally includes them
    drop_if_present = ["status_group", "id", "date_recorded", "year_recorded", "construction_year"]
    to_drop = [c for c in drop_if_present if c in batch_df.columns]
    if to_drop:
        batch_df = batch_df.drop(columns=to_drop)

    # Keep only required features (ignore extras)
    extra = [c for c in batch_df.columns if c not in REQUIRED_FEATURES]
    prepared = batch_df[REQUIRED_FEATURES].copy()

    if extra:
        msg = (
            "File accepted. Some additional columns will be ignored:\n- "
            + "\n- ".join(extra)
        )
    else:
        msg = "File accepted. The column format is correct."

    # Coerce dtypes to match pipeline expectations
    prepared = coerce_batch_dtypes(prepared)

    return True, msg, prepared
# >>> FIX END


# >>> FIX START: coerce_batch_dtypes() dirapikan (tidak ada unreachable code)
def coerce_batch_dtypes(prepared: pd.DataFrame) -> pd.DataFrame:
    """
    Make batch dataframe dtypes consistent with training expectations.
    Common fix: boolean values in categorical columns (e.g., public_meeting/permit)
    should be converted to string so OneHotEncoder won't see np.True_/np.False_ as new labels.
    """
    df = prepared.copy()

    # Force numeric columns to numeric (coerce errors to NaN)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categorical columns normalization
    for c in CAT_COLS:
        # Convert bool dtype -> string
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(str)
        else:
            # Normalize True/False values if mixed in object column
            df[c] = df[c].replace({True: "True", False: "False"})

        # Normalize as string and strip spaces
        df[c] = df[c].astype("string").str.strip()

        # Treat typical missing-string tokens as NA
        df[c] = df[c].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

        # Optional: fill missing category to avoid encoder issues (safer for production)
        df[c] = df[c].fillna("unknown")

    # Optional: handle numeric NaN (depends on how your pipeline was trained)
    # If your pipeline can handle NaN via imputer, you can keep NaN.
    # If not, you may fill with 0 or median (but must match training).
    # for c in NUM_COLS:
    #     df[c] = df[c].fillna(0)

    return df
# >>> FIX END


    # Keep only required features (ignore extras)
    extra = [c for c in batch_df.columns if c not in REQUIRED_FEATURES]
    prepared = batch_df[REQUIRED_FEATURES].copy()

    if extra:
        msg = (
            "File accepted. Some additional columns will be ignored:\n- "
            + "\n- ".join(extra)
        )
    else:
        msg = "File accepted. The column format is correct."

    return True, msg, prepared

# ================================
# PAGE: OVERVIEW
# ================================
def page_overview(df: pd.DataFrame):
    st.title("Tanzania Water Pump Status Prediction")

    st.markdown("""
    This application predicts the **operational status of waterpoints** in Tanzania into three classes:

    - `functional`
    - `functional needs repair`
    - `non functional`

    The model used is a **Random Forest** with a preprocessing pipeline (numerical + categorical),
    built from historical waterpoint data that includes technical, managerial, and geospatial features.

    **Application structure:**
    1. **Overview**: project summary and dataset snapshot.
    2. **EDA**: exploration of target distribution, numerical and categorical features, and spatial distribution.
    3. **Prediction**: waterpoint status prediction for **single input** or **multiple inputs** via **CSV** (batch).
    """)

    st.subheader("Dataset Snapshot")
    st.write("Number of rows & columns:", df.shape)
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Target Distribution (status_group)")
    st.caption("Target distribution helps to understand class proportions and potential class imbalance.")
    target_counts = df["status_group"].value_counts(normalize=True) * 100
    st.write(target_counts.round(2).astype(str) + " %")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df,
        x="status_group",
        order=df["status_group"].value_counts().index,
        ax=ax
    )
    ax.set_title("Distribution of status_group")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ================================
# PAGE: EDA
# ================================
def page_eda(df: pd.DataFrame):
    st.title("Exploratory Data Analysis")

    st.markdown("### 1) Summary Statistics of Numerical Features")
    st.dataframe(df[NUM_COLS].describe().T, use_container_width=True)

    st.markdown("### 2) Distribution of Numerical Features")
    selected_num = st.selectbox("Select a numerical feature:", NUM_COLS)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[selected_num], kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_num}")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("### 3) Categorical Features vs Target")
    st.caption("Note: Some features have many categories; the visualization may appear crowded for features with high cardinality.")
    selected_cat = st.selectbox("Select a categorical feature:", CAT_COLS)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    sns.countplot(data=df, x=selected_cat, hue="status_group", ax=ax2)
    plt.xticks(rotation=45, ha="right")
    ax2.set_title(f"{selected_cat} vs status_group")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown("### 4) Simple Map (Longitude vs Latitude)")
    st.caption("Scatter plot of waterpoint locations in Tanzania (sample)")
    sample_df = df.sample(min(5000, len(df)), random_state=42)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sample_df,
        x="longitude",
        y="latitude",
        hue="status_group",
        s=10,
        alpha=0.6,
        ax=ax3
    )
    ax3.set_title("Distribution of Waterpoint Locations (sample)")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

# ================================
# PAGE: PREDICTION
# ================================
def page_prediction(df: pd.DataFrame, model):
    st.title("Prediction")

    st.markdown("""
This page is used to predict the status of waterpoints in two ways:

### 1) Single Prediction
Suitable for predicting **one waterpoint** (e.g., input from a field survey).
**Output:** predicted class and the probability for each class.                

### 2) Batch Prediction (CSV)
Suitable for predicting **multiple waterpoints at once** using a CSV file.
**Output:** results table and a downloadable CSV file containing `predicted_status_group`.

**Data privacy:** uploaded files are processed only temporarily during the session and are **not stored** by the application.                                

""")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # -------------------------
    # Tab 1: Single Prediction
    # -------------------------
    with tab1:
        st.subheader("Single Waterpoint Prediction")

        st.info("""
**How to use (Single):**
1. Enter numerical values (e.g., `gps_height`, `longitude`, `latitude`, `population`).
2. Select categorical values (e.g., `basin`, `management`, `water_quality`, etc).
3. Click **Predict Status** to see the prediction and class probabilities.

**Interpretation tip:** If the highest probability is much larger than the others, the model is relatively more confident in its prediction.
""")

        col1, col2 = st.columns(2)

        with col1:
            amount_tsh = st.number_input("amount_tsh", min_value=0.0, value=0.0)
            gps_height = st.number_input("gps_height", value=0.0)
            longitude = st.number_input("longitude", value=35.0)
            latitude = st.number_input("latitude", value=-6.0)
            population = st.number_input("population", min_value=0.0, value=100.0)
            pump_age = st.number_input("pump_age (tahun)", min_value=0.0, value=10.0)

        with col2:
            basin = st.selectbox("basin", sorted(df["basin"].dropna().unique()))
            region_code = st.selectbox("region_code", sorted(df["region_code"].dropna().unique()))
            district_code = st.selectbox("district_code", sorted(df["district_code"].dropna().unique()))
            public_meeting = st.selectbox("public_meeting", sorted(df["public_meeting"].dropna().unique()))
            permit = st.selectbox("permit", sorted(df["permit"].dropna().unique()))
            extraction_type_class = st.selectbox("extraction_type_class", sorted(df["extraction_type_class"].dropna().unique()))
            management = st.selectbox("management", sorted(df["management"].dropna().unique()))
            payment_type = st.selectbox("payment_type", sorted(df["payment_type"].dropna().unique()))
            water_quality = st.selectbox("water_quality", sorted(df["water_quality"].dropna().unique()))
            quantity = st.selectbox("quantity", sorted(df["quantity"].dropna().unique()))
            source_type = st.selectbox("source_type", sorted(df["source_type"].dropna().unique()))
            source_class = st.selectbox("source_class", sorted(df["source_class"].dropna().unique()))
            waterpoint_type = st.selectbox("waterpoint_type", sorted(df["waterpoint_type"].dropna().unique()))

        if st.button("Predict Status"):
            input_dict = {
                "amount_tsh": [amount_tsh],
                "gps_height": [gps_height],
                "longitude": [longitude],
                "latitude": [latitude],
                "population": [population],
                "pump_age": [pump_age],
                "basin": [basin],
                "region_code": [region_code],
                "district_code": [district_code],
                "public_meeting": [public_meeting],
                "permit": [permit],
                "extraction_type_class": [extraction_type_class],
                "management": [management],
                "payment_type": [payment_type],
                "water_quality": [water_quality],
                "quantity": [quantity],
                "source_type": [source_type],
                "source_class": [source_class],
                "waterpoint_type": [waterpoint_type]
            }

            input_df = pd.DataFrame(input_dict)[REQUIRED_FEATURES]

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.markdown("### Prediction Result")
            st.success(f"Predicted status_group: **{pred}**")

            st.markdown("### Class Probabilities")
            proba_df = pd.DataFrame({
                "status_group": model.classes_,
                "probability": proba
            }).sort_values("probability", ascending=False)

            st.dataframe(proba_df, use_container_width=True)

    # -------------------------
    # Tab 2: Batch Prediction
    # -------------------------
    with tab2:
        st.subheader("Batch Prediction via CSV")

        st.info("""
**How to use (Batch):**
1. Prepare a CSV file (1 row = 1 waterpoint).
2. Ensure the file contains the feature columns used by the model and **does not include** the `status_group` column..
3. Upload the file → click **Predict Batch** → download the results.

Don’t have data yet? Download the **template** or **dummy CSV** below to try this feature.
""")

        # ---- Dummy / template downloads ----
        template_bytes = pd.DataFrame(columns=REQUIRED_FEATURES).to_csv(index=False).encode("utf-8")

        dummy_small = [
            {
                "amount_tsh": 0, "gps_height": 1390, "longitude": 34.9381, "latitude": -9.8563, "population": 109, "pump_age": 15,
                "basin": "Lake Nyasa", "region_code": 11, "district_code": 5, "public_meeting": True, "permit": False,
                "extraction_type_class": "gravity", "management": "vwc", "payment_type": "annually", "water_quality": "soft",
                "quantity": "enough", "source_type": "spring", "source_class": "groundwater", "waterpoint_type": "communal standpipe"
            },
            {
                "amount_tsh": 25, "gps_height": 686, "longitude": 37.4607, "latitude": -3.8213, "population": 250, "pump_age": 8,
                "basin": "Pangani", "region_code": 21, "district_code": 4, "public_meeting": False, "permit": True,
                "extraction_type_class": "handpump", "management": "company", "payment_type": "monthly", "water_quality": "fluoride",
                "quantity": "insufficient", "source_type": "shallow well", "source_class": "groundwater", "waterpoint_type": "hand pump"
            }
        ]

        dummy_realistic = [
            {
                "amount_tsh": 0, "gps_height": 1025, "longitude": 35.215, "latitude": -6.123, "population": 80, "pump_age": 20,
                "basin": "Internal Drainage", "region_code": 14, "district_code": 3, "public_meeting": True, "permit": False,
                "extraction_type_class": "gravity", "management": "vwc", "payment_type": "annually", "water_quality": "soft",
                "quantity": "insufficient", "source_type": "river", "source_class": "groundwater", "waterpoint_type": "communal standpipe"
            },
            {
                "amount_tsh": 10, "gps_height": 980, "longitude": 36.112, "latitude": -7.345, "population": 150, "pump_age": 5,
                "basin": "Lake Victoria", "region_code": 18, "district_code": 6, "public_meeting": True, "permit": True,
                "extraction_type_class": "handpump", "management": "water board", "payment_type": "monthly", "water_quality": "soft",
                "quantity": "enough", "source_type": "borehole", "source_class": "groundwater", "waterpoint_type": "hand pump"
            },
            {
                "amount_tsh": 0, "gps_height": 450, "longitude": 39.100, "latitude": -5.210, "population": 40, "pump_age": 30,
                "basin": "Rufiji", "region_code": 6, "district_code": 1, "public_meeting": False, "permit": False,
                "extraction_type_class": "gravity", "management": "other", "payment_type": "never", "water_quality": "salty",
                "quantity": "dry", "source_type": "dam", "source_class": "surface", "waterpoint_type": "communal standpipe"
            },
            {
                "amount_tsh": 50, "gps_height": 1200, "longitude": 34.700, "latitude": -8.950, "population": 300, "pump_age": 3,
                "basin": "Lake Nyasa", "region_code": 11, "district_code": 4, "public_meeting": True, "permit": True,
                "extraction_type_class": "submersible", "management": "company", "payment_type": "monthly", "water_quality": "soft",
                "quantity": "enough", "source_type": "borehole", "source_class": "groundwater", "waterpoint_type": "mechanized pump"
            },
            {
                "amount_tsh": 0, "gps_height": 670, "longitude": 37.820, "latitude": -3.900, "population": 90, "pump_age": 12,
                "basin": "Pangani", "region_code": 21, "district_code": 7, "public_meeting": False, "permit": False,
                "extraction_type_class": "handpump", "management": "vwc", "payment_type": "annually", "water_quality": "fluoride",
                "quantity": "insufficient", "source_type": "shallow well", "source_class": "groundwater", "waterpoint_type": "hand pump"
            }
        ]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.download_button(
                "Download Template CSV",
                data=template_bytes,
                file_name="batch_template.csv",
                mime="text/csv"
            )
        with col_b:
            st.download_button(
                "Download Dummy CSV (Small)",
                data=make_dummy_csv_bytes(dummy_small),
                file_name="dummy_batch_small.csv",
                mime="text/csv"
            )
        with col_c:
            st.download_button(
                "Download Dummy CSV (Realistic)",
                data=make_dummy_csv_bytes(dummy_realistic),
                file_name="dummy_batch_realistic.csv",
                mime="text/csv"
            )

        st.divider()

        st.markdown("**Columns used by the model:**")
        st.write(", ".join(REQUIRED_FEATURES))

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            batch_df_raw = pd.read_csv(uploaded_file)
            st.write("Preview data (raw):")
            st.dataframe(batch_df_raw.head(), use_container_width=True)

            ok, msg, batch_df = validate_and_prepare_batch_df(batch_df_raw)

            if ok:
                st.success(msg)
                st.write("Preview data (prepared for model):")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("Predict Batch"):
                    batch_pred = model.predict(batch_df)

                    result_df = batch_df_raw.copy()
                    result_df["predicted_status_group"] = batch_pred

                    st.write("Prediction results (preview):")
                    st.dataframe(result_df.head(), use_container_width=True)

                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download prediction results CSV",
                        data=csv_out,
                        file_name="batch_prediction_output.csv",
                        mime="text/csv"
                    )
            else:
                st.error(msg)

# ================================
# MAIN
# ================================
def main():
    df = load_data()
    model = load_model()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Overview", "EDA", "Prediction"])

    if page == "Overview":
        page_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Prediction":
        page_prediction(df, model)

if __name__ == "__main__":
    main()
