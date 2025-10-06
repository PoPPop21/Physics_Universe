import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import csv
import base64
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ======================
# CONFIGURACIÓN GLOBAL
# ======================
st.set_page_config(
    page_title="Explorador de Exoplanetas Habitables",
    layout="wide",
    page_icon="🪐"
)

# CSS personalizado
st.markdown("""
<style>
    /* Mejorar legibilidad global */
    .stApp {
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
    }
    
    /* Headers principales */
    .main-header {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        animation: fadeIn 1.5s;
        letter-spacing: 1.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Cards mejorados */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card h3 {
        font-size: 22px;
        font-weight: 700;
        margin: 15px 0 10px 0;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        letter-spacing: 0.5px;
    }
    
    .metric-card p {
        font-size: 15px;
        font-weight: 500;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
        line-height: 1.5;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    
    /* Scores de habitabilidad */
    .habitability-score {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.3);
    }
    
    .score-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .score-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Animaciones */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Botones mejorados */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.3s;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar mejorado */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.85);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stRadio label {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Mejorar contraste de texto en elementos de Streamlit */
    .stMarkdown {
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    
    /* Títulos de sección */
    h1, h2, h3 {
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Metricas de Streamlit */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 15px;
        font-weight: 600;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ======================
# FUNCIONES DE UTILIDAD
# ======================

def robust_read_csv(uploaded_file):
    """Lee CSV ignorando comentarios y detectando separador automáticamente."""
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
        return None, "El archivo está vacío o solo tiene comentarios."

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
        buf.seek(0)
        try:
            df = pd.read_csv(buf, sep=detected_sep, engine="python")
            if df.shape[1] > 1:
                return df, "Leído con delimitador detectado: " + str(detected_sep)
        except Exception:
            pass

    for sep in [",", ";", "\t", "|"]:
        buf.seek(0)
        try:
            df_temp = pd.read_csv(buf, sep=sep, engine="python")
            if df_temp.shape[1] > 1:
                return df_temp, "Leído con separador probado: " + str(sep)
        except Exception:
            continue

    buf.seek(0)
    try:
        df_try = pd.read_csv(buf, engine="python", on_bad_lines="skip")
        if df_try.shape[1] > 1:
            return df_try, "Leído con fallback on_bad_lines='skip'."
    except Exception as e:
        return None, f"Error final: {e}"

    return None, "No se detectó separador válido."


def calculate_habitability_score(row):
    """Calcula un score de habitabilidad basado en múltiples factores."""
    score = 0
    max_score = 100
    
    # Radio planetario (Tierra-like: 0.5 - 2.0 radios terrestres)
    if 'pl_rade' in row and pd.notna(row['pl_rade']):
        if 0.5 <= row['pl_rade'] <= 2.0:
            score += 20
        elif 0.3 <= row['pl_rade'] <= 3.0:
            score += 10
    
    # Temperatura de equilibrio (zona habitable: 200-320K)
    if 'pl_eqt' in row and pd.notna(row['pl_eqt']):
        if 200 <= row['pl_eqt'] <= 320:
            score += 25
        elif 150 <= row['pl_eqt'] <= 400:
            score += 15
    
    # Insolación (similar a la Tierra: 0.25 - 1.5)
    if 'pl_insol' in row and pd.notna(row['pl_insol']):
        if 0.25 <= row['pl_insol'] <= 1.5:
            score += 20
        elif 0.1 <= row['pl_insol'] <= 2.0:
            score += 10
    
    # Temperatura estelar (enanas tipo M-K: 2500-5500K)
    if 'st_teff' in row and pd.notna(row['st_teff']):
        if 2500 <= row['st_teff'] <= 5500:
            score += 20
        elif 2000 <= row['st_teff'] <= 6500:
            score += 10
    
    # Periodo orbital (no demasiado corto ni largo)
    if 'pl_orbper' in row and pd.notna(row['pl_orbper']):
        if 10 <= row['pl_orbper'] <= 500:
            score += 15
        elif 1 <= row['pl_orbper'] <= 1000:
            score += 7
    
    return score


def set_background(image_file, zoom_out=False):
    """Aplica una imagen de fondo con CSS en base64."""
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        
        # Configuración de fondo según si queremos zoom out
        if zoom_out:
            background_size = "100% auto"
            background_position = "center top"
        else:
            background_size = "cover"
            background_position = "center center"
        
        css = """
        <style>
        .stApp {
            background: url("data:image/png;base64,%s") no-repeat;
            background-size: %s;
            background-position: %s;
            background-attachment: fixed;
        }
        </style>
        """ % (encoded, background_size, background_position)
        st.markdown(css, unsafe_allow_html=True)
    except:
        pass


def create_example_dataset():
    """Crea un dataset de ejemplo más completo."""
    np.random.seed(42)
    n = 50
    
    data = {
        "pl_name": ["Exoplanet-" + str(i) for i in range(n)],
        "pl_rade": np.random.uniform(0.3, 5.0, n),
        "pl_orbper": np.random.uniform(1, 500, n),
        "pl_eqt": np.random.uniform(100, 800, n),
        "pl_insol": np.random.uniform(0.05, 5.0, n),
        "st_teff": np.random.uniform(2000, 7000, n),
        "st_rad": np.random.uniform(0.1, 2.0, n),
        "st_mass": np.random.uniform(0.1, 1.5, n),
        "pl_dens": np.random.uniform(0.5, 10.0, n),
    }
    
    df = pd.DataFrame(data)
    
    # Agregar algunos planetas conocidos habitables
    known_planets = pd.DataFrame({
        "pl_name": ["Kepler-186f", "TOI-700 d", "K2-18 b", "TRAPPIST-1 e"],
        "pl_rade": [1.11, 1.19, 2.7, 0.92],
        "pl_orbper": [129.9, 37.4, 33.0, 6.1],
        "pl_eqt": [188, 270, 300, 251],
        "pl_insol": [0.32, 0.86, 1.1, 0.66],
        "st_teff": [3755, 3480, 3500, 2566],
        "st_rad": [0.47, 0.42, 0.45, 0.117],
        "st_mass": [0.48, 0.42, 0.45, 0.089],
        "pl_dens": [5.5, 5.0, 3.0, 5.2],
    })
    
    df = pd.concat([known_planets, df], ignore_index=True)
    return df


# ======================
# PÁGINA INICIAL
# ======================
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

if st.session_state["page"] == "landing":
    image_path = Path(__file__).parent / "assets" / "interfaz.png"
    if image_path.exists():
        set_background(str(image_path), zoom_out=True)  # Sin zoom para ver completa

    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
    video_path = Path(__file__).parent / "assets" / "universo.mp4"
    if video_path.exists():
        st.video(str(video_path))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Logo NASA Space Apps en la esquina
    st.markdown("""
        <div style='position: absolute; top: 20px; left: 20px; z-index: 1000;'>
            <div style='background: rgba(255,255,255,0.15); 
                        backdrop-filter: blur(10px);
                        padding: 15px 25px; 
                        border-radius: 50px;
                        border: 2px solid rgba(255,255,255,0.3);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
                <p style='margin: 0; 
                          font-size: 18px; 
                          font-weight: 700; 
                          color: white;
                          text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
                          letter-spacing: 1px;'>
                    🚀 NASA SPACE APPS - ASTROEN
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <h1 style='font-size: 62px; color: #FFFFFF; font-weight: 900; 
                       text-shadow: 4px 4px 20px rgba(0,0,0,0.95), 
                                    0 0 30px rgba(102, 126, 234, 0.8),
                                    0 0 50px rgba(118, 75, 162, 0.6);
                       letter-spacing: 2px;
                       font-family: "Segoe UI", Arial, sans-serif;
                       margin-bottom: 25px;'>
                🌌 Explorador de Exoplanetas Habitables
            </h1>
            <p style='font-size: 24px; 
                      color: #FFFFFF; 
                      font-weight: 600;
                      text-shadow: 3px 3px 12px rgba(0,0,0,0.9),
                                   0 0 20px rgba(0,0,0,0.7); 
                      margin-top: 20px;
                      letter-spacing: 1px;
                      font-family: "Segoe UI", Arial, sans-serif;'>
                🚀 Un viaje interactivo hacia mundos lejanos con IA avanzada
            </p>
            <p style='font-size: 19px; 
                      color: #E8F0FF; 
                      font-weight: 500;
                      text-shadow: 2px 2px 10px rgba(0,0,0,0.9),
                                   0 0 15px rgba(0,0,0,0.6);
                      letter-spacing: 0.5px;
                      font-family: "Segoe UI", Arial, sans-serif;
                      line-height: 1.6;'>
                🤖 Predicción con Machine Learning | 📊 Visualizaciones 3D | 🌍 Score de Habitabilidad
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 INICIAR EXPLORACIÓN", key="start"):
            st.session_state["page"] = "main"
            st.rerun()

# ======================
# PÁGINA PRINCIPAL
# ======================
elif st.session_state["page"] == "main":
    image_path2 = Path(__file__).parent / "assets" / "interfaz2.png"
    if image_path2.exists():
        set_background(str(image_path2), zoom_out=False)  # Con zoom para llenar pantalla

    st.sidebar.markdown("## 🔭 Panel de Navegación")
    section = st.sidebar.radio(
        "Selecciona una sección:",
        ["🏠 Inicio", "📂 Exploración de Datos", "📊 Visualizaciones Avanzadas", 
         "🌍 Panel Habitabilidad", "🤖 Predicción con IA", "🔬 Análisis Comparativo"]
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Volver al Inicio"):
        st.session_state["page"] = "landing"
        st.rerun()

    # ======================
    # INICIO
    # ======================
    if section == "🏠 Inicio":
        st.markdown("<h1 class='main-header'>🌌 Dashboard Principal</h1>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h2>📂</h2>
                    <h3>Exploración</h3>
                    <p>Carga y analiza datasets de exoplanetas</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h2>📊</h2>
                    <h3>Visualización</h3>
                    <p>Gráficos interactivos en 2D y 3D</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h2>🌍</h2>
                    <h3>Habitabilidad</h3>
                    <p>Scores y métricas detalladas</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class='metric-card'>
                    <h2>🤖</h2>
                    <h3>IA Predictiva</h3>
                    <p>Machine Learning avanzado</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Instrucciones Rápidas")
        st.info("""
        **1.** Ve a '📂 Exploración de Datos' y carga tu CSV o usa el dataset de ejemplo.
        
        **2.** Explora las visualizaciones interactivas en '📊 Visualizaciones Avanzadas'.
        
        **3.** Consulta el '🌍 Panel Habitabilidad' para ver scores detallados.
        
        **4.** Entrena modelos de IA en '🤖 Predicción con IA' para predecir habitabilidad.
        """)

    # ======================
    # EXPLORACIÓN DE DATOS
    # ======================
    elif section == "📂 Exploración de Datos":
        st.markdown("<h1 class='main-header'>📂 Exploración de Datos</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("📁 Sube tu archivo CSV de exoplanetas", type=["csv"])
        
        with col2:
            use_example = st.button("🎲 Usar Dataset de Ejemplo", use_container_width=True)

        if use_example:
            df = create_example_dataset()
            st.success("✅ Dataset de ejemplo cargado con éxito!")
            st.session_state["df"] = df
            
        elif uploaded_file is not None:
            df, message = robust_read_csv(uploaded_file)
            if df is not None:
                st.success("✅ Dataset cargado con éxito!")
                st.info(message)
                st.session_state["df"] = df
            else:
                st.error("❌ Error al leer el CSV: " + str(message))
        else:
            st.info("👉 Sube un archivo CSV o usa el dataset de ejemplo para comenzar.")

        if "df" in st.session_state:
            df = st.session_state["df"]
            
            # Estadísticas generales
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Total Planetas", len(df))
            col2.metric("📋 Columnas", df.shape[1])
            missing_pct = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
            col3.metric("✅ Datos Completos", "{:.1%}".format(missing_pct))
            col4.metric("🔢 Numéricos", len(df.select_dtypes(include=['number']).columns))
            
            st.markdown("### 📋 Vista Previa del Dataset")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Información detallada
            with st.expander("ℹ️ Información Detallada del Dataset"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Tipos de Datos:**")
                    st.write(df.dtypes)
                with col2:
                    st.markdown("**Valores Faltantes:**")
                    st.write(df.isnull().sum())
            
            # Estadísticas descriptivas
            with st.expander("📊 Estadísticas Descriptivas"):
                st.write(df.describe())

    # ======================
    # VISUALIZACIONES AVANZADAS
    # ======================
    elif section == "📊 Visualizaciones Avanzadas":
        st.markdown("<h1 class='main-header'>📊 Visualizaciones Interactivas</h1>", unsafe_allow_html=True)
        
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset en la sección 'Exploración de Datos'")
        else:
            df = st.session_state["df"]
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("❌ Se necesitan al menos 2 columnas numéricas para visualizar.")
            else:
                viz_type = st.selectbox(
                    "🎨 Tipo de Visualización",
                    ["Scatter 2D", "Scatter 3D", "Matriz de Correlación", "Distribuciones", "Box Plots"]
                )
                
                if viz_type == "Scatter 2D":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x = st.selectbox("Eje X", numeric_cols, index=0)
                    with col2:
                        y = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols)-1))
                    with col3:
                        color_col = st.selectbox("Color por", ["Ninguno"] + numeric_cols)
                    
                    title_text = "Relación entre " + str(x) + " y " + str(y)
                    fig = px.scatter(
                        df, x=x, y=y,
                        color=color_col if color_col != "Ninguno" else None,
                        hover_data=df.columns.tolist(),
                        title=title_text,
                        template="plotly_dark"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Scatter 3D":
                    if len(numeric_cols) < 3:
                        st.error("Se necesitan al menos 3 columnas numéricas para 3D")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            x = st.selectbox("Eje X", numeric_cols, index=0)
                        with col2:
                            y = st.selectbox("Eje Y", numeric_cols, index=1)
                        with col3:
                            z = st.selectbox("Eje Z", numeric_cols, index=2)
                        with col4:
                            color_col = st.selectbox("Color", ["Ninguno"] + numeric_cols)
                        
                        title_text = "Visualización 3D: " + str(x) + " vs " + str(y) + " vs " + str(z)
                        fig = px.scatter_3d(
                            df, x=x, y=y, z=z,
                            color=color_col if color_col != "Ninguno" else None,
                            hover_data=df.columns.tolist(),
                            title=title_text,
                            template="plotly_dark"
                        )
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Matriz de Correlación":
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(
                        corr,
                        text_auto='.2f',
                        aspect="auto",
                        title="Matriz de Correlación",
                        color_continuous_scale="RdBu_r",
                        template="plotly_dark"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Distribuciones":
                    col = st.selectbox("Selecciona columna", numeric_cols)
                    title_text = "Distribución de " + str(col)
                    fig = px.histogram(
                        df, x=col,
                        nbins=30,
                        title=title_text,
                        template="plotly_dark"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Box Plots":
                    cols_to_plot = st.multiselect("Selecciona columnas", numeric_cols, default=numeric_cols[:3])
                    if cols_to_plot:
                        fig = go.Figure()
                        for col in cols_to_plot:
                            fig.add_trace(go.Box(y=df[col], name=col))
                        fig.update_layout(
                            title="Box Plots Comparativos",
                            template="plotly_dark",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # PANEL HABITABILIDAD
    # ======================
    elif section == "🌍 Panel Habitabilidad":
        st.markdown("<h1 class='main-header'>🌍 Panel de Habitabilidad</h1>", unsafe_allow_html=True)
        
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset")
        else:
            df = st.session_state["df"].copy()
            
            # Calcular scores
            df['habitability_score'] = df.apply(calculate_habitability_score, axis=1)
            df = df.sort_values('habitability_score', ascending=False)
            
            # Filtros
            st.markdown("### 🎛️ Filtros de Habitabilidad")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'pl_rade' in df.columns:
                    radius_range = st.slider("Radio Planetario (R⊕)", 0.0, 10.0, (0.5, 2.0))
            with col2:
                if 'pl_eqt' in df.columns:
                    temp_range = st.slider("Temperatura (K)", 0, 1000, (200, 320))
            with col3:
                min_score = st.slider("Score Mínimo", 0, 100, 50)
            
            # Aplicar filtros
            filtered_df = df[df['habitability_score'] >= min_score]
            if 'pl_rade' in df.columns:
                filtered_df = filtered_df[
                    (filtered_df['pl_rade'] >= radius_range[0]) & 
                    (filtered_df['pl_rade'] <= radius_range[1])
                ]
            if 'pl_eqt' in df.columns:
                filtered_df = filtered_df[
                    (filtered_df['pl_eqt'] >= temp_range[0]) & 
                    (filtered_df['pl_eqt'] <= temp_range[1])
                ]
            
            st.markdown("### 🎯 Planetas Encontrados: " + str(len(filtered_df)))
            
            # Top 3 candidatos
            st.markdown("### 🏆 Top 3 Candidatos Más Habitables")
            
            for idx, row in filtered_df.head(3).iterrows():
                score = row['habitability_score']
                score_class = "score-high" if score >= 70 else "score-medium" if score >= 50 else "score-low"
                
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        score_html = """
                            <div class='habitability-score %s'>
                                %d
                            </div>
                        """ % (score_class, score)
                        st.markdown(score_html, unsafe_allow_html=True)
                    with col2:
                        planet_name = row.get('pl_name', 'Planeta ' + str(idx))
                        st.markdown("#### " + str(planet_name))
                        cols = st.columns(4)
                        if 'pl_rade' in row:
                            cols[0].metric("Radio", "{:.2f} R⊕".format(row['pl_rade']))
                        if 'pl_eqt' in row:
                            cols[1].metric("Temp.", "{:.0f} K".format(row['pl_eqt']))
                        if 'pl_insol' in row:
                            cols[2].metric("Insolación", "{:.2f}".format(row['pl_insol']))
                        if 'st_teff' in row:
                            cols[3].metric("T Estelar", "{:.0f} K".format(row['st_teff']))
                st.markdown("---")
            
            # Tabla completa
            with st.expander("📋 Ver Todos los Candidatos"):
                st.dataframe(filtered_df, use_container_width=True)
            
            # Visualización de scores
            fig = px.bar(
                filtered_df.head(20),
                x='habitability_score',
                y='pl_name' if 'pl_name' in filtered_df.columns else filtered_df.index,
                orientation='h',
                title="Score de Habitabilidad - Top 20",
                template="plotly_dark",
                color='habitability_score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================
    # PREDICCIÓN CON IA
    # ======================
    elif section == "🤖 Predicción con IA":
        st.markdown("<h1 class='main-header'>🤖 Predicción con Machine Learning</h1>", unsafe_allow_html=True)
        
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset")
        else:
            df = st.session_state["df"].copy()
            
            st.markdown("### 🎯 Configuración del Modelo")
            
            # Preparar datos
            df['is_habitable'] = df.apply(calculate_habitability_score, axis=1) >= 60
            
            # Seleccionar features
            available_features = df.select_dtypes(include=['number']).columns.tolist()
            if 'habitability_score' in available_features:
                available_features.remove('habitability_score')
            if 'is_habitable' in available_features:
                available_features.remove('is_habitable')
            
            features = st.multiselect(
                "Selecciona características para entrenar",
                available_features,
                default=available_features[:5] if len(available_features) >= 5 else available_features
            )
            
            if len(features) < 2:
                st.error("❌ Selecciona al menos 2 características")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2)
                with col2:
                    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                
                if st.button("🚀 ENTRENAR TODOS LOS MODELOS", use_container_width=True):
                    with st.spinner("Entrenando múltiples modelos de IA..."):
                        # Preparar datos
                        X = df[features].fillna(df[features].median())
                        y = df['is_habitable']
                        
                        # Split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Escalar
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Definir modelos
                        modelos = {
                            'Regresión Logística': LogisticRegression(max_iter=1000),
                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                            'Support Vector Machine': SVC(probability=True, random_state=42),
                            'K-Nearest Neighbors': KNeighborsClassifier(),
                            'Decision Tree': DecisionTreeClassifier(random_state=42),
                            'Naive Bayes': GaussianNB(),
                            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
                        }
                        
                        resultados = {}
                        mejores_modelos = {}
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Entrenar cada modelo
                        for idx, (nombre, modelo) in enumerate(modelos.items()):
                            status_text.text("Entrenando: " + nombre + "...")
                            
                            # Entrenar
                            modelo.fit(X_train_scaled, y_train)
                            
                            # Cross-validation
                            scores = cross_val_score(modelo, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
                            
                            # Predicciones
                            y_pred = modelo.predict(X_test_scaled)
                            
                            # Métricas
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            
                            resultados[nombre] = {
                                'cv_mean': np.mean(scores),
                                'cv_std': np.std(scores),
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'y_pred': y_pred
                            }
                            
                            mejores_modelos[nombre] = modelo
                            
                            progress_bar.progress((idx + 1) / len(modelos))
                        
                        status_text.text("✅ ¡Entrenamiento completado!")
                        progress_bar.empty()
                        
                        # Guardar en session state
                        st.session_state['resultados'] = resultados
                        st.session_state['mejores_modelos'] = mejores_modelos
                        st.session_state['scaler'] = scaler
                        st.session_state['features'] = features
                        st.session_state['y_test'] = y_test
                        
                        st.success("✅ ¡Todos los modelos entrenados exitosamente!")
                
                # Mostrar resultados si existen
                if 'resultados' in st.session_state:
                    resultados = st.session_state['resultados']
                    
                    st.markdown("---")
                    st.markdown("### 📊 Comparación de Modelos")
                    
                    # Crear DataFrame de resultados
                    df_resultados = pd.DataFrame({
                        'Modelo': list(resultados.keys()),
                        'CV Mean': [r['cv_mean'] for r in resultados.values()],
                        'CV Std': [r['cv_std'] for r in resultados.values()],
                        'Accuracy': [r['accuracy'] for r in resultados.values()],
                        'Precision': [r['precision'] for r in resultados.values()],
                        'Recall': [r['recall'] for r in resultados.values()],
                        'F1-Score': [r['f1'] for r in resultados.values()]
                    }).sort_values('Accuracy', ascending=False)
                    
                    # Identificar mejor modelo
                    mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
                    mejor_accuracy = df_resultados.iloc[0]['Accuracy']
                    
                    st.success("🏆 Mejor Modelo: **" + mejor_modelo_nombre + "** con {:.2%} de accuracy".format(mejor_accuracy))
                    
                    # Tabla de resultados con formato mejorado
                    st.markdown("#### 📋 Resultados Completos")
                    
                    # Formatear la tabla manualmente
                    df_resultados_fmt = df_resultados.copy()
                    df_resultados_fmt['CV Mean'] = df_resultados_fmt['CV Mean'].apply(lambda x: "{:.2%}".format(x))
                    df_resultados_fmt['CV Std'] = df_resultados_fmt['CV Std'].apply(lambda x: "{:.2%}".format(x))
                    df_resultados_fmt['Accuracy'] = df_resultados_fmt['Accuracy'].apply(lambda x: "{:.2%}".format(x))
                    df_resultados_fmt['Precision'] = df_resultados_fmt['Precision'].apply(lambda x: "{:.2%}".format(x))
                    df_resultados_fmt['Recall'] = df_resultados_fmt['Recall'].apply(lambda x: "{:.2%}".format(x))
                    df_resultados_fmt['F1-Score'] = df_resultados_fmt['F1-Score'].apply(lambda x: "{:.2%}".format(x))
                    
                    st.dataframe(
                        df_resultados_fmt,
                        use_container_width=True
                    )
                    
                    # Heatmap de métricas
                    st.markdown("#### 🔥 Heatmap de Rendimiento")
                    
                    # Preparar datos para heatmap
                    metrics_for_heatmap = df_resultados[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=metrics_for_heatmap,
                        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        y=df_resultados['Modelo'].tolist(),
                        colorscale='RdYlGn',
                        text=np.round(metrics_for_heatmap, 3),
                        texttemplate='%{text:.2%}',
                        textfont={"size": 12},
                        colorbar=dict(title="Score")
                    ))
                    
                    fig_heatmap.update_layout(
                        title='Heatmap de Métricas por Modelo',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Gráfico de barras comparativo
                    st.markdown("### 📈 Visualización Comparativa")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_acc = px.bar(
                            df_resultados,
                            x='Accuracy',
                            y='Modelo',
                            orientation='h',
                            title='Accuracy por Modelo',
                            template='plotly_dark',
                            color='Accuracy',
                            color_continuous_scale='Viridis'
                        )
                        fig_acc.update_layout(height=400)
                        st.plotly_chart(fig_acc, use_container_width=True)
                    
                    with col2:
                        fig_f1 = px.bar(
                            df_resultados,
                            x='F1-Score',
                            y='Modelo',
                            orientation='h',
                            title='F1-Score por Modelo',
                            template='plotly_dark',
                            color='F1-Score',
                            color_continuous_scale='Plasma'
                        )
                        fig_f1.update_layout(height=400)
                        st.plotly_chart(fig_f1, use_container_width=True)
                    
                    # Métricas del mejor modelo
                    st.markdown("### 🎯 Análisis Detallado del Mejor Modelo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    mejor_resultado = resultados[mejor_modelo_nombre]
                    
                    col1.metric("Accuracy", "{:.2%}".format(mejor_resultado['accuracy']))
                    col2.metric("Precision", "{:.2%}".format(mejor_resultado['precision']))
                    col3.metric("Recall", "{:.2%}".format(mejor_resultado['recall']))
                    col4.metric("F1-Score", "{:.2%}".format(mejor_resultado['f1']))
                    
                    # Matriz de confusión del mejor modelo
                    st.markdown("### 🎯 Matriz de Confusión - " + mejor_modelo_nombre)
                    cm = confusion_matrix(st.session_state['y_test'], mejor_resultado['y_pred'])
                    
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicción", y="Real", color="Cantidad"),
                        x=['No Habitable', 'Habitable'],
                        y=['No Habitable', 'Habitable'],
                        text_auto=True,
                        color_continuous_scale='Blues',
                        template="plotly_dark"
                    )
                    fig_cm.update_layout(height=500)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Importancia de características (si aplica)
                    if mejor_modelo_nombre in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
                        st.markdown("### 🔍 Importancia de Características")
                        modelo = st.session_state['mejores_modelos'][mejor_modelo_nombre]
                        
                        importance_df = pd.DataFrame({
                            'Característica': features,
                            'Importancia': modelo.feature_importances_
                        }).sort_values('Importancia', ascending=False)
                        
                        fig_imp = px.bar(
                            importance_df,
                            x='Importancia',
                            y='Característica',
                            orientation='h',
                            title="Importancia de Características - " + mejor_modelo_nombre,
                            template="plotly_dark",
                            color='Importancia',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Cross-validation scores
                    st.markdown("### 📊 Cross-Validation Scores")
                    cv_data = pd.DataFrame({
                        'Modelo': list(resultados.keys()),
                        'CV Mean ± Std': ["{:.2%} ± {:.2%}".format(r['cv_mean'], r['cv_std']) for r in resultados.values()]
                    })
                    st.dataframe(cv_data, use_container_width=True)
                
                # Sección de predicción individual
                if 'mejores_modelos' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 🔮 Hacer Predicción Individual")
                    
                    # Selector de modelo
                    modelo_seleccionado = st.selectbox(
                        "Selecciona un modelo para predecir:",
                        list(st.session_state['mejores_modelos'].keys())
                    )
                    
                    st.info("Ingresa los valores de un nuevo exoplaneta para predecir su habitabilidad")
                    
                    input_data = {}
                    cols = st.columns(3)
                    for idx, feature in enumerate(st.session_state['features']):
                        with cols[idx % 3]:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            mean_val = float(df[feature].mean())
                            
                            input_data[feature] = st.number_input(
                                feature,
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key="input_" + str(feature)
                            )
                    
                    if st.button("🎯 PREDECIR HABITABILIDAD", use_container_width=True):
                        modelo = st.session_state['mejores_modelos'][modelo_seleccionado]
                        scaler = st.session_state['scaler']
                        
                        # Preparar input
                        input_df = pd.DataFrame([input_data])
                        input_scaled = scaler.transform(input_df)
                        
                        # Predecir
                        prediction = modelo.predict(input_scaled)[0]
                        
                        # Obtener probabilidades si el modelo lo soporta
                        if hasattr(modelo, 'predict_proba'):
                            probability = modelo.predict_proba(input_scaled)[0]
                        else:
                            probability = [0.5, 0.5]  # Valor por defecto
                        
                        # Mostrar resultado
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if prediction:
                                prob_text = "{:.1%}".format(probability[1])
                                result_html = """
                                    <div class='habitability-score score-high'>
                                        ✅ HABITABLE
                                        <br>
                                        <span style='font-size: 24px;'>
                                        Probabilidad: %s
                                        </span>
                                    </div>
                                """ % prob_text
                                st.markdown(result_html, unsafe_allow_html=True)
                                st.balloons()
                            else:
                                prob_text = "{:.1%}".format(probability[0])
                                result_html = """
                                    <div class='habitability-score score-low'>
                                        ❌ NO HABITABLE
                                        <br>
                                        <span style='font-size: 24px;'>
                                        Probabilidad: %s
                                        </span>
                                    </div>
                                """ % prob_text
                                st.markdown(result_html, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### 📊 Detalles")
                            st.write("**Modelo usado:** " + modelo_seleccionado)
                            resultado_texto = st.session_state['resultados'][modelo_seleccionado]
                            st.write("**Accuracy del modelo:** {:.2%}".format(resultado_texto['accuracy']))
                            st.write("**F1-Score:** {:.2%}".format(resultado_texto['f1']))

    # ======================
    # ANÁLISIS COMPARATIVO
    # ======================
    elif section == "🔬 Análisis Comparativo":
        st.markdown("<h1 class='main-header'>🔬 Análisis Comparativo</h1>", unsafe_allow_html=True)
        
        if "df" not in st.session_state:
            st.warning("⚠️ Primero carga un dataset")
        else:
            df = st.session_state["df"].copy()
            df['habitability_score'] = df.apply(calculate_habitability_score, axis=1)
            
            st.markdown("### 🌍 Comparación con la Tierra")
            
            # Valores de la Tierra
            earth_values = {
                'pl_rade': 1.0,
                'pl_eqt': 288,
                'pl_insol': 1.0,
                'st_teff': 5778,
                'pl_orbper': 365.25
            }
            
            # Seleccionar planetas para comparar
            if 'pl_name' in df.columns:
                planets_to_compare = st.multiselect(
                    "Selecciona planetas para comparar",
                    df['pl_name'].tolist(),
                    default=df.nlargest(3, 'habitability_score')['pl_name'].tolist()
                )
                
                if planets_to_compare:
                    comparison_df = df[df['pl_name'].isin(planets_to_compare)]
                    
                    # Gráfico de radar
                    st.markdown("### 📊 Gráfico de Radar Comparativo")
                    
                    features_to_compare = ['pl_rade', 'pl_eqt', 'pl_insol', 'st_teff']
                    features_available = [f for f in features_to_compare if f in comparison_df.columns]
                    
                    if len(features_available) >= 3:
                        fig = go.Figure()
                        
                        # Normalizar valores
                        for feature in features_available:
                            max_val = df[feature].max()
                            if max_val > 0:
                                comparison_df[feature + '_norm'] = comparison_df[feature] / max_val
                        
                        # Añadir Tierra
                        earth_r = [earth_values.get(f, 0) / df[f].max() for f in features_available]
                        fig.add_trace(go.Scatterpolar(
                            r=earth_r,
                            theta=features_available,
                            fill='toself',
                            name='Tierra',
                            line=dict(color='green', width=2)
                        ))
                        
                        # Añadir planetas seleccionados
                        colors = px.colors.qualitative.Set2
                        for idx, (_, planet) in enumerate(comparison_df.iterrows()):
                            planet_r = [planet[f + '_norm'] for f in features_available]
                            fig.add_trace(go.Scatterpolar(
                                r=planet_r,
                                theta=features_available,
                                fill='toself',
                                name=planet.get('pl_name', 'Planeta ' + str(idx)),
                                line=dict(color=colors[idx % len(colors)])
                            ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True,
                            template="plotly_dark",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla comparativa
                    st.markdown("### 📋 Tabla Comparativa Detallada")
                    
                    display_cols = ['pl_name', 'habitability_score'] + features_available
                    display_cols = [c for c in display_cols if c in comparison_df.columns]
                    
                    st.dataframe(
                        comparison_df[display_cols].style.highlight_max(
                            subset=['habitability_score'], color='lightgreen'
                        ),
                        use_container_width=True
                    )
                    
                    # Análisis estadístico
                    st.markdown("### 📈 Análisis Estadístico")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Similitud con la Tierra (%):**")
                        for _, planet in comparison_df.iterrows():
                            similarity = 0
                            count = 0
                            for feature in features_available:
                                if feature in earth_values and pd.notna(planet[feature]):
                                    earth_val = earth_values[feature]
                                    planet_val = planet[feature]
                                    diff = abs(planet_val - earth_val) / earth_val
                                    similarity += max(0, 1 - diff)
                                    count += 1
                            if count > 0:
                                similarity_pct = (similarity / count) * 100
                                planet_name = planet.get('pl_name', 'N/A')
                                st.write("🌍 **" + str(planet_name) + "**: {:.1f}%".format(similarity_pct))
                    
                    with col2:
                        st.markdown("**Características Destacadas:**")
                        for _, planet in comparison_df.iterrows():
                            highlights = []
                            if 'pl_rade' in planet and 0.5 <= planet['pl_rade'] <= 2.0:
                                highlights.append("✅ Radio similar a la Tierra")
                            if 'pl_eqt' in planet and 200 <= planet['pl_eqt'] <= 320:
                                highlights.append("✅ Temperatura adecuada")
                            if 'pl_insol' in planet and 0.25 <= planet['pl_insol'] <= 1.5:
                                highlights.append("✅ Insolación apropiada")
                            
                            planet_name = planet.get('pl_name', 'N/A')
                            st.write("**" + str(planet_name) + "**:")
                            for h in highlights:
                                st.write("  " + h)
                            if not highlights:
                                st.write("  ⚠️ Condiciones extremas")
            else:
                st.info("El dataset no tiene columna 'pl_name' para comparar planetas individuales")
                
                # Análisis general sin nombres
                st.markdown("### 📊 Distribución General")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Selecciona una característica", numeric_cols)
                    
                    title_text = "Distribución de " + str(selected_col)
                    fig = px.histogram(
                        df,
                        x=selected_col,
                        nbins=30,
                        title=title_text,
                        template="plotly_dark",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p> Explorador de Exoplanetas Habitables | NASA Space Apps Challenge 2025</p>
    <p>Desarrollado usando Streamlit, Plotly y Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
