# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import io
import csv
import time

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Exoplanet Explorer", layout="wide", page_icon="🪐")

# --------------------------
# Helper: lectura robusta de CSV (quita comentarios y detecta delimitador)
# --------------------------
def robust_read_csv(uploaded_file) -> (pd.DataFrame | None, str):
    try:
        raw_bytes = uploaded_file.getvalue()
    except Exception as e:
        return None, f"Error leyendo bytes del archivo: {e}"

    try:
        text = raw_bytes.decode("utf-8")
    except Exception:
        try:
            text = raw_bytes.decode("latin-1")
        except Exception:
            text = raw_bytes.decode("utf-8", errors="replace")

    lines = text.splitlines()
    cleaned_lines = [ln for ln in lines if not ln.strip().startswith("#")]
    if len(cleaned_lines) == 0:
        return None, "El archivo solo contiene líneas comentadas o está vacío después de eliminar comentarios."

    cleaned_text = "\n".join(cleaned_lines)
    buf = io.StringIO(cleaned_text)

    detected_sep = None
    try:
        sample = "\n".join(cleaned_lines[:50])
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        detected_sep = dialect.delimiter
    except Exception:
        detected_sep = None

    if detected_sep:
        try:
            buf.seek(0)
            df = pd.read_csv(buf, sep=detected_sep, engine="python")
            if df.shape[1] > 1:
                return df, f"Leído con delimitador detectado: '{detected_sep}'"
        except Exception:
            pass

    for sep in [",", ";", "\t", "|"]:
        try:
            buf.seek(0)
            df_temp = pd.read_csv(buf, sep=sep, engine="python")
            if df_temp.shape[1] > 1:
                return df_temp, f"Leído con separador probado: '{sep}'"
        except Exception:
            continue

    try:
        buf.seek(0)
        df_try = pd.read_csv(buf, engine="python", on_bad_lines="skip")
        if df_try is not None and df_try.shape[1] > 1:
            return df_try, "Leído con fallback on_bad_lines='skip'."
        else:
            return None, "No se detectó separador válido."
    except Exception as e:
        return None, f"Error final: {e}"


# --------------------------
# LANDING PAGE
# --------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if st.session_state["page"] == "landing":
    try:
        st.video("assets/universo.mp4")
    except Exception:
        st.warning("No se encontró 'assets/universo.mp4'.")

    st.markdown(
        """
        <div style='text-align: center; margin-top: -50px;'>
            <h1 style='font-size: 48px; color: white; text-shadow: 2px 2px 12px rgba(0,0,0,0.9);'>
                🌌 Explorador de Exoplanetas Habitables
            </h1>
            <p style='font-size:18px; color:#e6eef6'>
                Un viaje interactivo hacia mundos lejanos — sube tu dataset o usa el de ejemplo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Entrar al Explorador"):
        st.session_state["page"] = "main"
        st.rerun()

# --------------------------
# APP PRINCIPAL
# --------------------------
elif st.session_state["page"] == "main":
    st.sidebar.title("🔭 Navegación")
    section = st.sidebar.radio("Ir a:", ["Exploración de Datos", "Visualizaciones", "Panel Habitabilidad", "Volver al inicio"])

    if section == "Volver al inicio":
        st.session_state["page"] = "landing"
        st.rerun()

    # --------------------------
    # Exploración de Datos
    # --------------------------
    if section == "Exploración de Datos":
        st.title("📂 Exploración de datos")
        uploaded_file = st.file_uploader("Sube tu archivo CSV de exoplanetas", type=["csv"])
        use_example = st.button("Usar dataset de ejemplo (pequeño)")

        if use_example:
            df = pd.DataFrame({
                "pl_name": ["Kepler-186f", "TOI-700 d", "K2-18 b"],
                "pl_rade": [1.11, 1.19, 2.7],
                "pl_orbper": [129.9, 37.4, 33.0],
                "pl_eqt": [188, 270, 300],
                "pl_insol": [0.32, 0.86, 1.1],
                "st_teff": [3755, 3480, 3500],
                "st_rad": [0.47, 0.42, 0.45]
            })
            st.success("Cargado dataset de ejemplo.")
            st.dataframe(df.head())
            st.session_state["df"] = df
        elif uploaded_file is not None:
            df, message = robust_read_csv(uploaded_file)
            if df is not None:
                st.success("✅ Dataset cargado con éxito.")
                st.info(message)
                st.dataframe(df.head(10))
                st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
                st.session_state["df"] = df
            else:
                st.error(f"❌ No se logró leer el CSV: {message}")
        else:
            st.info("👉 Sube un archivo CSV o usa el dataset de ejemplo.")

    # --------------------------
    # Visualizaciones
    # --------------------------
    elif section == "Visualizaciones":
        st.title("📊 Visualizaciones Interactivas")
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset.")
        else:
            df = st.session_state["df"]
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) < 1:
                st.error("No hay columnas numéricas para graficar.")
            else:
                x = st.selectbox("Eje X", numeric_cols, index=0)
                y = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols)-1))
                fig = px.scatter(df, x=x, y=y, hover_data=df.columns.tolist())
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Panel de Habitabilidad
    # --------------------------
    elif section == "Panel Habitabilidad":
        st.title("🌍 Panel de Habitabilidad")
        st.write("Filtros simples de habitabilidad.")

        rad_min, rad_max = st.slider("Radio planetario (R⊕)", 0.1, 10.0, (0.5, 2.5))
        teq_min, teq_max = st.slider("Temperatura equilibrio (K)", 100, 600, (200, 330))
        insol_min, insol_max = st.slider("Insolación (S⊕)", 0.0, 10.0, (0.2, 2.0))

        if "df" not in st.session_state:
            st.info("Carga un dataset primero.")
        else:
            df = st.session_state["df"]
            col_prad = next((c for c in ["pl_rade", "planet_radius", "prad"] if c in df.columns), None)
            col_teq = next((c for c in ["pl_eqt", "eq_temp", "teq"] if c in df.columns), None)
            col_insol = next((c for c in ["pl_insol", "insolation"] if c in df.columns), None)

            if not any([col_prad, col_teq, col_insol]):
                st.warning("No se encontraron columnas estándar.")
            else:
                dfc = df.copy()
                if col_prad: dfc[col_prad] = pd.to_numeric(dfc[col_prad], errors="coerce")
                if col_teq: dfc[col_teq] = pd.to_numeric(dfc[col_teq], errors="coerce")
                if col_insol: dfc[col_insol] = pd.to_numeric(dfc[col_insol], errors="coerce")

                mask = pd.Series([True]*len(dfc))
                if col_prad: mask &= dfc[col_prad].between(rad_min, rad_max)
                if col_teq: mask &= dfc[col_teq].between(teq_min, teq_max)
                if col_insol: mask &= dfc[col_insol].between(insol_min, insol_max)

                candidatos = dfc[mask]
                st.success(f"Se encontraron {len(candidatos)} candidatos.")
                if len(candidatos) > 0:
                    st.dataframe(candidatos.head(50))

