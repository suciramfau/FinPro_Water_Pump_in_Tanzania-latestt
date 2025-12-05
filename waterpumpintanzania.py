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
# HALAMAN: OVERVIEW
# ================================
def page_overview(df):
    st.title("💧 Tanzania Water Pump Prediction Project")

    st.markdown("""
    Proyek ini bertujuan untuk **memprediksi kondisi waterpoint** di Tanzania:
    - `functional`
    - `functional needs repair`
    - `non functional`

    Model Machine Learning yang digunakan adalah **Random Forest (tuned)**  
    yang dibangun dari data historis waterpoint dengan fitur teknis, manajerial, dan geospasial.

    Aplikasi ini memiliki 2 bagian utama:
    1. **EDA** – untuk eksplorasi data dan insight.
    2. **Prediction** – untuk memprediksi status waterpoint baru (single dan batch).
    """)

    st.subheader("Dataset Snapshot")
    st.write("Jumlah baris & kolom:", df.shape)
    st.dataframe(df.head())

    st.subheader("Distribusi Target (status_group)")
    target_counts = df["status_group"].value_counts(normalize=True) * 100
    st.write(target_counts.round(2).astype(str) + " %")

    fig, ax = plt.subplots(figsize=(5,4))
    sns.countplot(
        data=df,
        x="status_group",
        order=df["status_group"].value_counts().index,
        ax=ax
    )
    ax.set_title("Distribusi status_group")
    plt.xticks(rotation=20)
    st.pyplot(fig)

# ================================
# HALAMAN: EDA
# ================================
def page_eda(df):
    st.title("📊 Exploratory Data Analysis")

    # --- Ringkasan numerik ---
    st.markdown("### 1. Ringkasan Statistik Fitur Numerik")
    num_cols = ["amount_tsh", "gps_height", "longitude", "latitude",
                "population", "pump_age"]
    st.dataframe(df[num_cols].describe().T)

    # --- Distribusi fitur numerik ---
    st.markdown("### 2. Distribusi Fitur Numerik")
    selected_num = st.selectbox("Pilih fitur numerik:", num_cols)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[selected_num], kde=True, ax=ax)
    ax.set_title(f"Distribusi {selected_num}")
    st.pyplot(fig)

    # --- Kategorikal vs target ---
    st.markdown("### 3. Distribusi Fitur Kategorikal vs Target")
    cat_cols = [
        "basin",
        "region_code",
        "district_code",
        "public_meeting",
        "permit",
        "extraction_type_class",
        "management",
        "payment_type",
        "water_quality",
        "quantity",
        "source_type",
        "source_class",
        "waterpoint_type"
    ]

    selected_cat = st.selectbox("Pilih fitur kategorikal:", cat_cols)
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.countplot(data=df, x=selected_cat, hue="status_group", ax=ax2)
    plt.xticks(rotation=45)
    ax2.set_title(f"{selected_cat} vs status_group")
    st.pyplot(fig2)

    # --- Peta sederhana ---
    st.markdown("### 4. Peta Sederhana (Longitude vs Latitude)")
    st.caption("Scatter plot lokasi waterpoint di Tanzania (sample).")

    sample_df = df.sample(min(5000, len(df)), random_state=42)

    fig3, ax3 = plt.subplots(figsize=(6,5))
    sns.scatterplot(
        data=sample_df,
        x="longitude",
        y="latitude",
        hue="status_group",
        s=10,
        alpha=0.6,
        ax=ax3
    )
    ax3.set_title("Sebaran Lokasi Waterpoint (sample)")
    st.pyplot(fig3)

# ================================
# HALAMAN: PREDICTION
# ================================
def page_prediction(df, model):
    st.title("🔮 Predict Waterpoint Status")

    st.markdown("""
    Kamu bisa:
    1. Mengisi **form 1 baris** untuk prediksi single waterpoint, atau  
    2. Mengunggah **file CSV** untuk prediksi batch.
    """)

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # -------------------------
    # Tab 1: Single Prediction
    # -------------------------
    with tab1:
        st.subheader("Single Waterpoint Prediction")

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
            public_meeting=st.selectbox("public_meeting",sorted(df['public_meeting'].dropna().unique()))
            permit=st.selectbox("permit",sorted(df['permit'].dropna().unique()))
            extraction_type_class = st.selectbox("extraction_type_class", sorted(df["extraction_type_class"].unique()))
            management = st.selectbox("management", sorted(df["management"].unique()))
            payment_type = st.selectbox("payment_type", sorted(df["payment_type"].unique()))
            water_quality = st.selectbox("water_quality", sorted(df["water_quality"].unique()))
            quantity = st.selectbox("quantity", sorted(df["quantity"].unique()))
            source_type = st.selectbox("source_type", sorted(df["source_type"].unique()))
            source_class = st.selectbox("source_class", sorted(df["source_class"].unique()))
            waterpoint_type = st.selectbox("waterpoint_type", sorted(df["waterpoint_type"].unique()))

        if st.button("Predict Status"):
            # Susun menjadi DataFrame 1 baris
            input_dict = {
                "amount_tsh": [amount_tsh],
                "gps_height": [gps_height],
                "longitude": [longitude],
                "latitude": [latitude],
                "basin": [basin],
                "region_code": [region_code],
                "district_code": [district_code],
                "population": [population],
                "public_meeting": [public_meeting],
                "permit": [permit],
                "extraction_type_class": [extraction_type_class],
                "management": [management],
                "payment_type": [payment_type],
                "water_quality": [water_quality],
                "quantity": [quantity],
                "source_type": [source_type],
                "source_class": [source_class],
                "waterpoint_type": [waterpoint_type],
                "pump_age": [pump_age]
            }

            input_df = pd.DataFrame(input_dict)

            # Prediksi
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.markdown(f"### Predicted status_group: **{pred}**")
            st.write("Probabilitas per kelas:")
            proba_df = pd.DataFrame({
                "status_group": model.classes_,
                "probability": proba
            })
            st.dataframe(proba_df)

    # -------------------------
    # Tab 2: Batch Prediction
    # -------------------------
    with tab2:
        st.subheader("Batch Prediction via CSV")

        st.markdown("""
        **Catatan:**  
        - File CSV *sebaiknya* hanya berisi kolom fitur (tanpa `status_group`).  
        - Nama kolom harus sama dengan fitur yang digunakan model.
        """)

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview data:")
            st.dataframe(batch_df.head())

            if st.button("Predict Batch"):
                batch_pred = model.predict(batch_df)
                result_df = batch_df.copy()
                result_df["predicted_status_group"] = batch_pred

                st.write("Hasil prediksi:")
                st.dataframe(result_df.head())

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download hasil prediksi CSV",
                    data=csv_out,
                    file_name="batch_prediction_output.csv",
                    mime="text/csv"
                )

# ================================
# MAIN
# ================================
def main():
    df = load_data()
    model = load_model()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "EDA", "Prediction"]
    )

    if page == "Overview":
        page_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Prediction":
        page_prediction(df, model)


if __name__ == "__main__":
    main()
