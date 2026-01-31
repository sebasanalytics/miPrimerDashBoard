import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from scipy.stats import skew, kurtosis

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
        # treat as boolean if only two unique values or dtype is bool
        if df[col].dtype == bool or len(unique_vals) <= 2:
            bool_cols.append(col)
    return bool_cols

def detect_id_columns(df):
    id_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            id_cols.append(col)
    return id_cols


@st.cache_data
def compute_column_summary(df):
    rows = []
    for col in df.columns:
        col_data = df[col]
        n = len(col_data)
        n_missing = int(col_data.isnull().sum())
        pct_missing = n_missing / n if n else 0
        n_unique = int(col_data.nunique(dropna=True))
        dtype = str(col_data.dtype)

        stats = {
            "Columna": col,
            "Tipo": dtype,
            "Nulos": n_missing,
            "% Nulos": round(pct_missing, 3),
            "Únicos": n_unique,
        }

        if pd.api.types.is_numeric_dtype(col_data):
            arr = col_data.dropna().astype(float)
            if len(arr) > 0:
                iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
                stats.update({
                    "Media": float(arr.mean()),
                    "Mediana": float(np.median(arr)),
                    "Std": float(arr.std()),
                    "Min": float(arr.min()),
                    "Max": float(arr.max()),
                    "Skew": float(skew(arr)),
                    "Kurtosis": float(kurtosis(arr)),
                    "IQR": iqr,
                })
        rows.append(stats)

    return pd.DataFrame(rows)


def try_import_requests():
    try:
        import requests as _r
        return True
    except Exception:
        return False

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
    # Resumen estadístico enriquecido
    # ==========================
    with st.expander("📋 Resumen estadístico avanzado", expanded=False):
        summary_df = compute_column_summary(df)
        st.dataframe(summary_df.sort_values(["Tipo", "Columna"]).reset_index(drop=True))
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resumen (CSV)", csv, "summary.csv", "text/csv")

    # ==========================
    # Numéricas
    # ==========================
    if numeric_cols:
        st.header("📊 Variables numéricas")

        num_col = st.selectbox("Selecciona una variable para detalle", numeric_cols)

        col_data = filtered_df[num_col].dropna().astype(float)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_hist = px.histogram(
                filtered_df,
                x=num_col,
                nbins=50,
                title=f"Distribución de {num_col}",
                template="plotly_dark",
                marginal="box"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_kde = px.violin(
                filtered_df,
                y=num_col,
                box=True,
                points="all",
                title=f"Violin / KDE de {num_col}",
                template="plotly_dark"
            )
            st.plotly_chart(fig_kde, use_container_width=True)

        with c2:
            if len(col_data) > 0:
                st.metric("Media", float(col_data.mean()))
                st.metric("Mediana", float(col_data.median()))
                st.metric("Std", float(col_data.std()))
                st.metric("Skew", float(skew(col_data)))

        # Scatter matrix for a small set of numerics
        if len(numeric_cols) <= 10:
            st.subheader("Matriz de dispersión (scatter matrix)")
            fig_mat = px.scatter_matrix(
                filtered_df[numeric_cols].dropna(),
                dimensions=numeric_cols,
                title="Scatter matrix",
                template="plotly_dark"
            )
            st.plotly_chart(fig_mat, use_container_width=True)

    # ==========================
    # Categóricas
    # ==========================
    if categorical_cols:
        st.header("🏷️ Variables categóricas")

        cat_col = st.selectbox("Selecciona una categoría", categorical_cols)

        freq = filtered_df[cat_col].value_counts().reset_index()
        freq.columns = [cat_col, "Frecuencia"]

        fig_bar = px.bar(
            freq,
            x=cat_col,
            y="Frecuencia",
            title=f"Distribución de {cat_col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if len(freq) <= 10:
            fig_pie = px.pie(
                freq,
                names=cat_col,
                values="Frecuencia",
                title=f"Porcentaje de {cat_col}",
                template="plotly_dark"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Top categorías")
        st.table(freq.head(10))

    # ==========================
    # Correlaciones
    # ==========================
    if len(numeric_cols) > 1:
        st.header("🔗 Correlación")

        corr = filtered_df[numeric_cols].corr(method="pearson")

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Matriz de correlación (Pearson)",
            color_continuous_scale="RdBu_r",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Spearman
        corr_s = filtered_df[numeric_cols].corr(method="spearman")
        fig_s = px.imshow(
            corr_s,
            text_auto=True,
            aspect="auto",
            title="Matriz de correlación (Spearman)",
            color_continuous_scale="RdBu_r",
            template="plotly_dark"
        )
        st.plotly_chart(fig_s, use_container_width=True)

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

        agg = st.selectbox("Agrupar por", ["Año", "Mes", "Día"], index=0)
        if agg == "Año":
            filtered_df["__period__"] = filtered_df[date_col].dt.to_period("Y").dt.to_timestamp()
        elif agg == "Mes":
            filtered_df["__period__"] = filtered_df[date_col].dt.to_period("M").dt.to_timestamp()
        else:
            filtered_df["__period__"] = filtered_df[date_col].dt.to_period("D").dt.to_timestamp()

        ts = (
            filtered_df.groupby("__period__")
            .size()
            .reset_index(name="Conteo")
        )

        fig = px.line(
            ts,
            x="__period__",
            y="Conteo",
            markers=True,
            title=f"Evolución temporal ({date_col}) - {agg}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # allow aggregation of a numeric column
        if numeric_cols:
            num_for_ts = st.selectbox("Numérica para agregar (opcional)", [None] + numeric_cols)
            if num_for_ts:
                ts_num = (
                    filtered_df.dropna(subset=[date_col, num_for_ts])
                    .groupby("__period__")[num_for_ts]
                    .agg(["mean", "sum", "median"])
                    .reset_index()
                )
                st.plotly_chart(px.line(ts_num, x="__period__", y="mean", title=f"Media de {num_for_ts} por {agg}", template="plotly_dark"), use_container_width=True)

    # ==========================
    # Integración Groq (sidebar)
    # ==========================
    with st.sidebar.expander("🔌 Integración Groq", expanded=False):
        st.write("Conecta con la API de Groq (sólo prueba de conexión).")
        api_key = st.text_input("API Key de Groq", type="password")
        test_conn = st.button("Probar conexión a Groq")

        if test_conn:
            if not api_key:
                st.error("Proporciona una API Key.")
            elif not try_import_requests():
                st.error("La librería 'requests' no está instalada en el entorno.")
            else:
                try:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    resp = requests.get("https://api.groq.ai/v1", headers=headers, timeout=8)
                    if resp.status_code == 200:
                        st.success("Conexión exitosa (status 200)")
                    else:
                        st.warning(f"Respuesta recibida: {resp.status_code} — {resp.text[:200]}")
                except Exception as e:
                    st.error(f"Error al conectar: {e}")

else:
    st.info("👆 Sube un archivo CSV para iniciar el EDA")
