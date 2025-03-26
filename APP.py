import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Correlaciones Bienestar y Salud Mental", layout="wide")

st.title("Visualización Interactiva de Datos")
st.markdown("""
Esta aplicación permite explorar las correlaciones entre:
- **Conflicto Trabajo-Familia**
- **Burnout e Indicadores Laborales** (compromiso, intención de retiro y satisfacción)
- **Efectos Colaterales** (alienación, desgaste y somatización)
""")

# Cargar la base de datos desde el repositorio local
data_path = "cleaned_data.csv"
df = pd.read_csv(data_path)

st.subheader("Vista Previa de los Datos")
st.dataframe(df.head())

# Lista de columnas del DataFrame
all_columns = df.columns.tolist()

# Panel lateral para selección de variables
st.sidebar.header("Selección de Variables")
selected_vars = st.sidebar.multiselect(
    "Selecciona las variables para el análisis de correlaciones",
    options=all_columns,
    help="Elige las columnas que consideres relevantes para el análisis."
)

if len(selected_vars) < 2:
    st.warning("Selecciona al menos dos variables para poder calcular las correlaciones.")
else:
    # Cálculo de la matriz de correlaciones
    corr_matrix = df[selected_vars].corr()

    st.subheader("Matriz de Correlaciones")
    st.dataframe(corr_matrix.style.format("{:.2f}"))

    # Heatmap de la matriz de correlaciones
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Gráficos Interactivos")
    st.markdown("Selecciona dos variables para visualizar su relación con un scatter plot interactivo.")

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable en Eje X", selected_vars)
    with col2:
        y_var = st.selectbox("Variable en Eje Y", selected_vars, index=1 if len(selected_vars) > 1 else 0)

    # Gráfico interactivo con Plotly y línea de tendencia
    fig2 = px.scatter(df, x=x_var, y=y_var, trendline="ols", 
                      title=f"Relación entre {x_var} y {y_var}")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("## Indicadores Laborales y Efectos Colaterales")

# Sección: Indicadores Laborales
st.markdown("### Indicadores Laborales")
# Se define una búsqueda de columnas basadas en palabras clave
laboral_cols = [col for col in all_columns if any(keyword in col.lower() for keyword in ['compromiso', 'intención', 'satisfacción'])]
if laboral_cols:
    st.markdown("**Estadísticas descriptivas de los Indicadores Laborales:**")
    st.dataframe(df[laboral_cols].describe())
else:
    st.info("No se encontraron variables que coincidan con 'compromiso', 'intención' o 'satisfacción' para indicadores laborales.")

# Sección: Efectos Colaterales
st.markdown("### Efectos Colaterales")
efectos_cols = [col for col in all_columns if any(keyword in col.lower() for keyword in ['alienación', 'desgaste', 'somatización'])]
if efectos_cols:
    st.markdown("**Estadísticas descriptivas de los Efectos Colaterales:**")
    st.dataframe(df[efectos_cols].describe())
else:
    st.info("No se encontraron variables que coincidan con 'alienación', 'desgaste' o 'somatización' para efectos colaterales.")
