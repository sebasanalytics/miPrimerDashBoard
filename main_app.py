import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="EDA Universal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 EDA Universal Interactivo")
st.caption("Funciona con cualquier CSV — sin supuestos")

# ==========================
# Utilidades
# ==========================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def detect_datetime_columns(df):
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / len(df) > 0.6:
                    datetime_cols.append(col)
            except Exception:
                pass
    return datetime_cols

def detect_boolean_columns(df):
    bool_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2:
            bool_cols.append(col)
    return bool_cols

def detect_id_columns(df):
    id_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            id_cols.append(col)
    return id_cols

# ==========================
# Upload
# ==========================
uploaded_file = st.file_uploader(
    "📂 Sube cualquier archivo CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = load_data(uploaded_file)

    st.success("Archivo cargado correctamente")

    # ==========================
    # Detección de tipos
    # ==========================
    datetime_cols = detect_datetime_columns(df)
    boolean_cols = detect_boolean_columns(df)
    id_cols = detect_id_columns(df)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    categorical_cols = [
        c for c in categorical_cols
        if c not in datetime_cols and c not in boolean_cols
    ]

    # ==========================
    # KPIs
    # ==========================
    st.subheader("📌 Resumen general")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Nulos", int(df.isnull().sum().sum()))
    c4.metric("Duplicados", int(df.duplicated().sum()))

    # ==========================
    # Vista previa
    # ==========================
    with st.expander("👀 Vista previa del dataset"):
        st.dataframe(df.head(30))

    # ==========================
    # Sidebar: filtros
    # ==========================
    st.sidebar.header("🎛️ Filtros dinámicos")
    filtered_df = df.copy()

    for col in categorical_cols:
        options = filtered_df[col].dropna().unique()
        if len(options) > 1 and len(options) < 50:
            selected = st.sidebar.multiselect(
                col,
                options=options,
                default=options
            )
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # ==========================
    # Nulos
    # ==========================
    st.header("🧹 Valores nulos")

    null_df = (
        df.isnull()
        .sum()
        .reset_index()
        .rename(columns={"index": "Columna", 0: "Nulos"})
    )

    fig = px.bar(
        null_df,
        x="Columna",
        y="Nulos",
        title="Valores nulos por columna",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Numéricas
    # ==========================
    if numeric_cols:
        st.header("📊 Variables numéricas")

        num_col = st.selectbox("Selecciona una variable", numeric_cols)

        fig = px.histogram(
            filtered_df,
            x=num_col,
            marginal="box",
            nbins=40,
            title=f"Distribución de {num_col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Categóricas
    # ==========================
    if categorical_cols:
        st.header("🏷️ Variables categóricas")

        cat_col = st.selectbox("Selecciona una categoría", categorical_cols)

        freq = filtered_df[cat_col].value_counts().reset_index()
        freq.columns = [cat_col, "Frecuencia"]

        fig = px.bar(
            freq,
            x=cat_col,
            y="Frecuencia",
            title=f"Distribución de {cat_col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Correlaciones
    # ==========================
    if len(numeric_cols) > 1:
        st.header("🔗 Correlación")

        corr = filtered_df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Matriz de correlación",
            color_continuous_scale="RdBu_r",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Temporal
    # ==========================
    if datetime_cols:
        st.header("⏳ Análisis temporal")

        date_col = st.selectbox("Columna temporal", datetime_cols)
        filtered_df[date_col] = pd.to_datetime(
            filtered_df[date_col],
            errors="coerce"
        )

        filtered_df["__year__"] = filtered_df[date_col].dt.year

        ts = (
            filtered_df.groupby("__year__")
            .size()
            .reset_index(name="Conteo")
        )

        fig = px.line(
            ts,
            x="__year__",
            y="Conteo",
            markers=True,
            title=f"Evolución temporal ({date_col})",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Sube un archivo CSV para iniciar el EDA")
