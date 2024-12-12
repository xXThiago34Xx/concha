from shiny import App, ui, render
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

# Cargar datos
def load_data():
    data = pd.read_csv("df_merged.csv")
    return data

# Preprocesar datos
def preprocess_data(data):
    label_encoders = {}
    for col in ['cod_alumno', 'cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data, data_scaled, label_encoders

# Cargar y procesar datos
df = load_data()
df, df_scaled, label_encoders = preprocess_data(df)

# UI
app_ui = ui.page_fluid(
    ui.h2("Modelos Predictivos con Shiny para Python"),
    ui.input_select("model_type", "Selecciona un modelo", ["Regresión Lineal", "Random Forest", "Clustering"]),
    ui.input_select("target", "Selecciona el objetivo", ["val_calific_final", "num_rep"]),
    ui.input_slider("n_estimators", "Número de árboles (Random Forest)", 10, 200, 100, step=10),
    ui.input_slider("n_clusters", "Número de clústeres (Clustering)", 2, 10, 3),
    ui.output_text("output_summary"),
    ui.output_plot("output_plot")  # Salida para gráficos
)

# Server
def server(input, output, session):
    @output
    @render.text
    def output_summary():
        if input.model_type() == "Regresión Lineal":
            model = LinearRegression()
            target = input.target()
            y = df[target]
            X = df[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
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
            return f"Modelo: Random Forest\nError Cuadrático Medio (MSE): {mse:.2f}"
        
        elif input.model_type() == "Clustering":
            model = KMeans(n_clusters=input.n_clusters(), random_state=42)
            model.fit(df_scaled[['cod_facultad', 'cod_escuela', 'cod_plan', 'cod_asignatura']])
            df['Cluster'] = model.labels_
            return f"Modelo: Clustering\nSe generaron {input.n_clusters()} clústeres."

    @output
    @render.plot(alt="Gráfico generado por el modelo")
    def output_plot():
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if input.model_type() == "Regresión Lineal":
            target = input.target()
            y = df[target]
            X = df['cod_facultad']
            ax.scatter(X, y, color="blue", label="Datos")
            ax.set_title("Regresión Lineal: cod_facultad vs " + target)
            ax.set_xlabel("cod_facultad")
            ax.set_ylabel(target)

        elif input.model_type() == "Random Forest":
            target = input.target()
            y = df[target]
            X = df['cod_facultad']
            ax.scatter(X, y, color="green", label="Datos")
            ax.set_title("Random Forest: cod_facultad vs " + target)
            ax.set_xlabel("cod_facultad")
            ax.set_ylabel(target)

        elif input.model_type() == "Clustering":
            ax.scatter(df['cod_facultad'], df['val_calific_final'], c=df['Cluster'], cmap='viridis')
            ax.set_title("Clustering basado en Facultad y Calificaciones")
            ax.set_xlabel("cod_facultad")
            ax.set_ylabel("val_calific_final")
        
        return fig

# Aplicación Shiny
app = App(app_ui, server)
