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
- **Burnout e Indicadores Laborales** (compromiso, intención de retiro, satisfacción y síntomas de burnout)
- **Efectos Colaterales** (alienación, desgaste y somatización)
""")

# Cargar la base de datos desde el repositorio local
data_path = "cleaned_data.csv"
df = pd.read_csv(data_path)

st.subheader("Vista Previa de los Datos")
st.dataframe(df.head())

# Lista de columnas del DataFrame
all_columns = df.columns.tolist()

# Panel lateral para selección de variables (a nivel ítem)
st.sidebar.header("Selección de Variables (Ítems)")
selected_vars = st.sidebar.multiselect(
    "Selecciona las variables para el análisis de correlaciones a nivel de ítems",
    options=all_columns,
    help="Elige las columnas que consideres relevantes para el análisis."
)

if len(selected_vars) < 2:
    st.warning("Selecciona al menos dos variables para poder calcular las correlaciones a nivel de ítems.")
else:
    # Cálculo de la matriz de correlaciones para los ítems
    corr_matrix = df[selected_vars].corr()

    st.subheader("Matriz de Correlaciones (Ítems)")
    st.dataframe(corr_matrix.style.format("{:.2f}"))

    # Heatmap de la matriz de correlaciones
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Gráficos Interactivos (Ítems)")
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

# Sección: Indicadores Laborales (búsqueda por palabras clave)
st.markdown("### Indicadores Laborales")
laboral_cols = [col for col in all_columns if any(keyword in col.lower() for keyword in ['compromiso', 'intención', 'satisfacción'])]
if laboral_cols:
    st.markdown("**Estadísticas descriptivas de los Indicadores Laborales:**")
    st.dataframe(df[laboral_cols].describe())
else:
    st.info("No se encontraron variables que coincidan con 'compromiso', 'intención' o 'satisfacción' para indicadores laborales.")

# Sección: Efectos Colaterales (búsqueda por palabras clave)
st.markdown("### Efectos Colaterales")
efectos_cols = [col for col in all_columns if any(keyword in col.lower() for keyword in ['alienación', 'desgaste', 'somatización'])]
if efectos_cols:
    st.markdown("**Estadísticas descriptivas de los Efectos Colaterales:**")
    st.dataframe(df[efectos_cols].describe())
else:
    st.info("No se encontraron variables que coincidan con 'alienación', 'desgaste' o 'somatización' para efectos colaterales.")

st.markdown("## Correlaciones por Dimensiones")

st.markdown("""
A continuación se agrupan los ítems por dimensiones a partir de los prefijos de cada columna. Se han definido tres dimensiones:
- **Conflicto Trabajo-Familia:** columnas que contienen el prefijo `(FT)`
- **Burnout e Indicadores Laborales:** columnas que contienen alguno de los prefijos `(CP)`, `(ST)`, `(IR)` o `(SB)`
- **Efectos Colaterales:** columnas que contienen alguno de los prefijos `(CS)`, `(CD)` o `(CA)`
""")

# Definir las dimensiones agrupando columnas según los prefijos indicados
dimensiones = {
    "Conflicto Trabajo-Familia": [col for col in all_columns if "(FT)" in col],
    "Burnout e Indicadores Laborales": [col for col in all_columns if any(prefijo in col for prefijo in ["(CP)", "(ST)", "(IR)", "(SB)"])],
    "Efectos Colaterales": [col for col in all_columns if any(prefijo in col for prefijo in ["(CS)", "(CD)", "(CA)"])]
}

# Mostrar las columnas encontradas por dimensión
for dim, cols in dimensiones.items():
    if cols:
        st.markdown(f"**{dim}**: Se encontraron {len(cols)} ítems.")
    else:
        st.info(f"No se encontraron ítems para la dimensión: {dim}")

# Calcular puntaje (media) por dimensión para cada registro, usando solo columnas numéricas
puntajes = pd.DataFrame()
for dim, cols in dimensiones.items():
    if cols:
        # Seleccionar únicamente las columnas numéricas para evitar errores
        numeric_cols = df[cols].select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            puntajes[dim] = df[numeric_cols].mean(axis=1)
        else:
            st.info(f"Para la dimensión {dim} no se encontraron columnas numéricas.")

st.markdown("### Vista de Puntajes por Dimensión")
st.dataframe(puntajes.head())

# Selección de dimensiones para análisis de correlación (por defecto se muestran todas)
dimensiones_seleccionadas = st.multiselect(
    "Selecciona las dimensiones para el análisis de correlación",
    options=puntajes.columns.tolist(),
    default=puntajes.columns.tolist(),
    help="Elige las dimensiones que deseas analizar a nivel de puntajes."
)

if len(dimensiones_seleccionadas) < 2:
    st.warning("Selecciona al menos dos dimensiones para poder calcular las correlaciones.")
else:
    # Calcular la matriz de correlaciones entre las dimensiones seleccionadas
    corr_dim = puntajes[dimensiones_seleccionadas].corr()

    st.subheader("Matriz de Correlaciones entre Dimensiones")
    st.dataframe(corr_dim.style.format("{:.2f}"))

    # Heatmap para las correlaciones de dimensiones
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_dim, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Gráfico Interactivo entre Dimensiones")
    st.markdown("Selecciona dos dimensiones para visualizar su relación con un scatter plot interactivo.")

    col_dim1, col_dim2 = st.columns(2)
    with col_dim1:
        x_dim = st.selectbox("Dimensión en Eje X", dimensiones_seleccionadas)
    with col_dim2:
        y_dim = st.selectbox("Dimensión en Eje Y", dimensiones_seleccionadas, index=1 if len(dimensiones_seleccionadas) > 1 else 0)

    fig4 = px.scatter(puntajes, x=x_dim, y=y_dim, trendline="ols",
                      title=f"Relación entre {x_dim} y {y_dim}")
    st.plotly_chart(fig4, use_container_width=True)
