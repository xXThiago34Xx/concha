from shiny import App, ui, render
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
def load_data():
    df_merged = pd.read_csv("df_merged.csv")
    df_repitencias = pd.read_csv("df_repitencias.csv")
    df_calificaciones_avg = pd.read_csv("df_calificaciones_avg.csv")
    return df_merged, df_repitencias, df_calificaciones_avg

# Preprocesar datos
def preprocess_data(data):
    label_encoders = {}
    for col in ['cod_alumno', 'cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    return data, label_encoders

# Cargar y procesar datos
df_merged, df_repitencias, df_calificaciones_avg = load_data()
df_merged, label_encoders = preprocess_data(df_merged)

df_repitencias = df_repitencias.fillna("N/A").astype(str)
df_calificaciones_avg = df_calificaciones_avg.fillna("N/A").astype(str)

# Convertir valores máximos y mínimos a tipos básicos de Python
cod_facultad_max = int(df_merged['cod_facultad'].max())
cod_escuela_max = int(df_merged['cod_escuela'].max())
cod_plan_max = int(df_merged['cod_plan'].max())
cod_asignatura_max = int(df_merged['cod_asignatura'].max())

# UI
app_ui = ui.page_fillable(
    ui.navset_card_pill(
        # Primera pestaña: Modelos
        ui.nav_panel(
            "Modelos",
            ui.h2("Modelos Predictivos"),
            ui.input_select("model_type", "Selecciona un modelo", ["Regresión Lineal", "Random Forest", "Clustering"]),
            ui.input_select("target", "Selecciona el objetivo", ["val_calific_final", "num_rep"]),
            ui.input_slider("n_estimators", "Número de árboles (Random Forest)", 10, 200, 100, step=10),
            ui.input_slider("n_clusters", "Número de clústeres (Clustering)", 2, 10, 3),
            ui.output_text("output_summary"),
            ui.output_plot("output_plot")
        ),
        # Segunda pestaña: Predicción
        ui.nav_panel(
            "Predicción",
            ui.h2("Realizar Predicción"),
            ui.input_select("predict_model", "Selecciona un modelo", ["Regresión Lineal", "Random Forest", "Clustering"]),
            ui.input_select("predict_target", "Selecciona el objetivo", ["val_calific_final", "num_rep"]),
            ui.input_numeric("input_cod_facultad", "Código de Facultad", 0, min=0, max=cod_facultad_max),
            ui.input_numeric("input_cod_escuela", "Código de Escuela", 0, min=0, max=cod_escuela_max),
            ui.input_numeric("input_cod_plan", "Código de Plan", 0, min=0, max=cod_plan_max),
            ui.input_numeric("input_cod_asignatura", "Código de Asignatura", 0, min=0, max=cod_asignatura_max),
            ui.output_text("prediction_result")
        ),
        # Tercera pestaña: Visualización de repitencias
        ui.nav_panel(
            "Repitencias",
            ui.h2("Datos de Repitencias"),
            ui.output_table("table_repitencias"),
            ui.output_ui("pagination_controls_repitencias")
        ),
        ui.nav_panel(
            "Calificaciones",
            ui.h2("Datos de Calificaciones"),
            ui.output_table("table_calificaciones_avg"),
            ui.output_ui("pagination_controls_calificaciones")
        ),
        id="main_tab"  # ID del conjunto de pestañas
    )
)

# Server
def server(input, output, session):

    # Tamaño de página configurable
    page_size = 10

    @output
    @render.text
    def output_summary():
        if input.model_type() == "Regresión Lineal":
            model = LinearRegression()
            target = input.target()
            y = df_merged[target]
            X = df_merged[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']]
            model.fit(X, y)
            mse = mean_squared_error(y, model.predict(X))
            return f"Modelo: Regresión Lineal\nError Cuadrático Medio (MSE): {mse:.2f}"
        
        elif input.model_type() == "Random Forest":
            model = RandomForestRegressor(n_estimators=input.n_estimators(), random_state=42)
            target = input.target()
            y = df_merged[target]
            X = df_merged[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']]
            model.fit(X, y)
            mse = mean_squared_error(y, model.predict(X))
            return f"Modelo: Random Forest\nError Cuadrático Medio (MSE): {mse:.2f}"
        
        elif input.model_type() == "Clustering":
            model = KMeans(n_clusters=input.n_clusters(), random_state=42)
            model.fit(df_merged[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']])
            df_merged['Cluster'] = model.labels_
            return f"Modelo: Clustering\nSe generaron {input.n_clusters()} clústeres."

    @output
    @render.plot(alt="Gráfico generado por el modelo")
    def output_plot():
        fig, ax = plt.subplots(figsize=(8, 6))

        if input.model_type() == "Regresión Lineal":
            target = input.target()
            y = df_merged[target]
            X = df_merged['cod_facultad']
            ax.scatter(X, y, color="blue", label="Datos")
            ax.set_title("Regresión Lineal: cod_facultad vs " + target)
            ax.set_xlabel("cod_facultad")
            ax.set_ylabel(target)
        
        elif input.model_type() == "Random Forest":
            target = input.target()
            y = df_merged[target]
            X = df_merged['cod_facultad']
            ax.scatter(X, y, color="green", label="Datos")
            ax.set_title("Random Forest: cod_facultad vs " + target)
            ax.set_xlabel("cod_facultad")
            ax.set_ylabel(target)
        
        elif input.model_type() == "Clustering":
            ax.scatter(df_merged['cod_facultad'], df_merged['val_calific_final'], c=df_merged['Cluster'], cmap='viridis')
            ax.set_title("Clustering basado en Facultad y Calificaciones")
            ax.set_xlabel("cod_facultad")
            ax.set_ylabel("val_calific_final")

        return fig

    @output
    @render.text
    def prediction_result():
        model_type = input.predict_model()
        target = input.predict_target()
        
        if model_type == "Clustering":
            return "El modelo Clustering no admite predicciones para nuevos datos."

        X = df_merged[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']]
        y = df_merged[target]

        if model_type == "Regresión Lineal":
            model = LinearRegression()
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=input.n_estimators(), random_state=42)
        else:
            return "Selecciona un modelo válido."

        model.fit(X, y)

        input_data = np.array([[input.input_cod_facultad(), input.input_cod_escuela(),
                                input.input_cod_plan(), input.input_cod_asignatura()]])
        prediction = model.predict(input_data)[0]

        return f"Predicción ({target}): {prediction:.2f}"
        
    @output
    @render.table
    def table_repitencias():
        # Obtener el número de página actual
        page = int(input.page_repitencias()) if "page_repitencias" in input else 1
        start_index = (page - 1) * page_size
        end_index = start_index + page_size

        # Filtrar datos para la página actual
        return df_repitencias.iloc[start_index:end_index]

    @output
    @render.table
    def table_calificaciones_avg():
        # Obtener el número de página actual
        page = int(input.page_calificaciones()) if "page_calificaciones" in input else 1
        start_index = (page - 1) * page_size
        end_index = start_index + page_size

        # Filtrar datos para la página actual
        return df_calificaciones_avg.iloc[start_index:end_index]

    @output
    @render.ui
    def pagination_controls_repitencias():
        total_pages_repitencias = (len(df_repitencias) + page_size - 1) // page_size
        return ui.input_numeric("page_repitencias", "Página (Repitencias)", value=1, min=1, max=total_pages_repitencias)

    @output
    @render.ui
    def pagination_controls_calificaciones():
        total_pages_calificaciones = (len(df_calificaciones_avg) + page_size - 1) // page_size
        return ui.input_numeric("page_calificaciones", "Página (Calificaciones)", value=1, min=1, max=total_pages_calificaciones)


# Aplicación Shiny
app = App(app_ui, server)
