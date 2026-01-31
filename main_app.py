import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="EDA Energías Renovables",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ EDA Interactivo – Energías Renovables")

# ======================
# Carga del archivo
# ======================
uploaded_file = st.file_uploader(
    "📂 Sube un archivo CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Archivo cargado correctamente ✅")

    # ======================
    # KPIs
    # ======================
    st.subheader("📌 Métricas generales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Proyectos", df.shape[0])
    col2.metric("Variables", df.shape[1])
    col3.metric("Nulos", int(df.isnull().sum().sum()))

    if "Capacidad_Instalada_MW" in df.columns:
        col4.metric(
            "Capacidad total (MW)",
            round(df["Capacidad_Instalada_MW"].sum(), 2)
        )

    # ======================
    # Vista previa
    # ======================
    with st.expander("👀 Vista previa del dataset"):
        st.dataframe(df.head(20))

    # ======================
    # Filtros dinámicos
    # ======================
    st.sidebar.header("🎛️ Filtros")

    filtered_df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        selected = st.sidebar.multiselect(
            f"Filtrar por {col}",
            options=df[col].unique(),
            default=df[col].unique()
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # ======================
    # Distribución numérica
    # ======================
    st.header("📊 Distribución de variables numéricas")

    numeric_cols = filtered_df.select_dtypes(include=["int64", "float64"]).columns

    selected_num_col = st.selectbox(
        "Selecciona una variable numérica",
        numeric_cols
    )

    fig = px.histogram(
        filtered_df,
        x=selected_num_col,
        nbins=30,
        marginal="box",
        title=f"Distribución de {selected_num_col}",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # Comparaciones categóricas
    # ======================
    st.header("🏷️ Comparaciones por categoría")

    if len(numeric_cols) > 0:
        cat_col = st.selectbox(
            "Selecciona una variable categórica",
            df.select_dtypes(include="object").columns
        )

        fig = px.box(
            filtered_df,
            x=cat_col,
            y=selected_num_col,
            color=cat_col,
            title=f"{selected_num_col} por {cat_col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # Correlación
    # ======================
    st.header("🔗 Correlación entre variables")

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

    # ======================
    # Análisis temporal
    # ======================
    if "Fecha_Entrada_Operacion" in filtered_df.columns:
        st.header("⏳ Evolución temporal")

        filtered_df["Fecha_Entrada_Operacion"] = pd.to_datetime(
            filtered_df["Fecha_Entrada_Operacion"],
            errors="coerce"
        )
        filtered_df["Año"] = filtered_df["Fecha_Entrada_Operacion"].dt.year

        time_series = (
            filtered_df
            .groupby("Año")["Capacidad_Instalada_MW"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            time_series,
            x="Año",
            y="Capacidad_Instalada_MW",
            markers=True,
            title="Capacidad instalada a lo largo del tiempo",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Sube un archivo CSV para comenzar el análisis")
