import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Energías Renovables", layout="wide")

st.title("📊 Análisis Exploratorio de Datos - Energías Renovables")

# ======================
# Carga del archivo
# ======================
uploaded_file = st.file_uploader(
    "Sube el archivo CSV a analizar",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Archivo cargado correctamente ✅")

    # ======================
    # Vista general
    # ======================
    st.header("📌 Vista general del dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores nulos", df.isnull().sum().sum())

    st.subheader("Primeras filas")
    st.dataframe(df.head())

    # ======================
    # Tipos de datos
    # ======================
    st.subheader("Tipos de datos")
    st.dataframe(df.dtypes.astype(str))

    # ======================
    # Estadísticas descriptivas
    # ======================
    st.subheader("Estadísticas descriptivas (numéricas)")
    st.dataframe(df.describe())

    # ======================
    # Variables categóricas
    # ======================
    st.header("📊 Análisis de variables categóricas")

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        st.subheader(f"{col}")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    # ======================
    # Variables numéricas
    # ======================
    st.header("📈 Análisis de variables numéricas")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        st.subheader(col)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df[col], kde=True, ax=ax[0])
        ax[0].set_title("Distribución")

        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title("Boxplot")

        st.pyplot(fig)

    # ======================
    # Correlaciones
    # ======================
    st.header("🔗 Matriz de correlación")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    # ======================
    # Análisis temporal
    # ======================
    if "Fecha_Entrada_Operacion" in df.columns:
        st.header("⏳ Análisis temporal")

        df["Fecha_Entrada_Operacion"] = pd.to_datetime(
            df["Fecha_Entrada_Operacion"],
            errors="coerce"
        )
        df["Año"] = df["Fecha_Entrada_Operacion"].dt.year

        proyectos_por_año = df.groupby("Año").size()

        fig, ax = plt.subplots()
        proyectos_por_año.plot(ax=ax)
        ax.set_ylabel("Número de proyectos")
        ax.set_title("Proyectos por año")
        st.pyplot(fig)

else:
    st.info("👆 Sube un archivo CSV para comenzar el análisis")
