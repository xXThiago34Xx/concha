from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import os

# Cargar datos
def load_data():
    data = pd.read_csv("df_merged.csv")
    return data

# Preprocesar datos
def preprocess_data(data):
    # Convertir columnas categóricas a numéricas
    label_encoders = {}
    for col in ['cod_alumno', 'cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Escalado opcional (para modelos como clustering)
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data, data_scaled, label_encoders

# Cargar y procesar datos
df = load_data()
df, df_scaled, label_encoders = preprocess_data(df)

# Crear carpeta para guardar gráficos
os.makedirs("plots", exist_ok=True)

# UI
app_ui = ui.page_fluid(
    ui.h2("Modelos Predictivos con Shiny para Python"),
    ui.input_select("model_type", "Selecciona un modelo", ["Regresión Lineal", "Random Forest", "Clustering"]),
    ui.input_select("target", "Selecciona el objetivo", ["val_calific_final", "num_rep"]),
    ui.input_slider("n_estimators", "Número de árboles (Random Forest)", 10, 200, 100, step=10),
    ui.input_slider("n_clusters", "Número de clústeres (Clustering)", 2, 10, 3),
    ui.output_text("output_summary"),
    ui.output_image("output_plot")
)

# Server
def server(input, output, session):
    # Estado reactivo para controlar la finalización del procesamiento
    model_state = reactive.Value("idle")  # Puede ser "processing", "done", o "idle"

    @output
    @render.text
    def output_summary():
        # Cambiar el estado a "processing" al iniciar el procesamiento
        model_state.set("processing")

        if input.model_type() == "Regresión Lineal":
            model = LinearRegression()
            target = input.target()
            y = df[target]
            X = df[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            # Cambiar el estado a "done" después de procesar
            model_state.set("done")
            return f"Modelo: Regresión Lineal\nError Cuadrático Medio (MSE): {mse:.2f}"
        
        elif input.model_type() == "Random Forest":
            model = RandomForestRegressor(n_estimators=input.n_estimators(), random_state=42)
            target = input.target()
            y = df[target]
            X = df[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            # Cambiar el estado a "done" después de procesar
            model_state.set("done")
            return f"Modelo: Random Forest\nError Cuadrático Medio (MSE): {mse:.2f}"
        
        elif input.model_type() == "Clustering":
            model = KMeans(n_clusters=input.n_clusters(), random_state=42)
            model.fit(df_scaled[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']])
            df['Cluster'] = model.labels_
            # Cambiar el estado a "done" después de procesar
            model_state.set("done")
            return f"Modelo: Clustering\nSe generaron {input.n_clusters()} clústeres."

    @output
    @render.image
    def output_plot():
        # Esperar hasta que el modelo termine de procesar
        if model_state.get() != "done":
            return {"src": None, "alt": "Procesando... Por favor, espera."}

        filepath = None

        if input.model_type() == "Regresión Lineal":
            # Crear gráfico para Regresión Lineal
            plt.figure(figsize=(8, 6))
            target = input.target()
            y = df[target]
            X = df['cod_facultad']
            plt.scatter(X, y, color="blue", label="Datos")
            plt.title("Regresión Lineal: cod_facultad vs " + target)
            plt.xlabel("cod_facultad")
            plt.ylabel(target)
            filepath = "plots/linear_regression_plot.png"
            plt.savefig(filepath)
            plt.close()

        elif input.model_type() == "Random Forest":
            # Crear gráfico para Random Forest
            plt.figure(figsize=(8, 6))
            target = input.target()
            y = df[target]
            X = df['cod_facultad']
            plt.scatter(X, y, color="green", label="Datos")
            plt.title("Random Forest: cod_facultad vs " + target)
            plt.xlabel("cod_facultad")
            plt.ylabel(target)
            filepath = "plots/random_forest_plot.png"
            plt.savefig(filepath)
            plt.close()

        elif input.model_type() == "Clustering":
            # Crear gráfico para Clustering
            plt.figure(figsize=(8, 6))
            plt.scatter(df['cod_facultad'], df['val_calific_final'], c=df['Cluster'], cmap='viridis')
            plt.title("Clustering basado en Facultad y Calificaciones")
            plt.xlabel("cod_facultad")
            plt.ylabel("val_calific_final")
            filepath = "plots/clustering_plot.png"
            plt.savefig(filepath)
            plt.close()

        # Validar que la imagen fue creada antes de devolverla
        if filepath and os.path.exists(filepath):
            return {"src": filepath, "alt": f"Gráfico para {input.model_type()}"}
        else:
            return {"src": None, "alt": "Imagen no disponible"}

# Aplicación Shiny
app = App(app_ui, server)