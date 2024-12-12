import pandas as pd

# Mapa de códigos de facultad a nombres
facultades = {
    1: "MEDICINA",
    2: "DERECHO Y CIENCIA POLÍTICA",
    3: "LETRAS Y CIENCIAS HUMANAS",
    4: "FARMACIA Y BIOQUÍMICA",
    5: "ODONTOLOGÍA",
    6: "EDUCACIÓN",
    7: "QUÍMICA E INGENIERÍA QUÍMICA",
    8: "MEDICINA VETERINARIA",
    9: "CIENCIAS ADMINISTRATIVAS",
    10: "CIENCIAS BIOLÓGICAS",
    11: "CIENCIAS CONTABLES",
    12: "CIENCIAS ECONÓMICAS",
    13: "CIENCIAS FÍSICAS",
    14: "CIENCIAS MATEMÁTICAS",
    15: "CIENCIAS SOCIALES",
    16: "INGENIERÍA GEOLÓGICA, MINERA, METALÚRGICA Y GEOGRÁFICA",
    17: "INGENIERÍA INDUSTRIAL",
    18: "PSICOLOGÍA",
    19: "INGENIERÍA ELECTRÓNICA Y ELÉCTRICA",
    20: "INGENIERÍA DE SISTEMAS E INFORMÁTICA"
}

# Leer las dos hojas del archivo Excel
def procesar_excel_y_crear_csv(ruta_excel, ruta_csv_salida):
    # Leer las hojas
    hoja1 = pd.read_excel(ruta_excel, sheet_name=0)
    hoja2 = pd.read_excel(ruta_excel, sheet_name=1)

    # Combinar las hojas
    datos_combinados = pd.concat([hoja1, hoja2], ignore_index=True)

    # Reemplazar cod_facultad con NombreFacultad
    datos_combinados['NombreFacultad'] = datos_combinados['cod_facultad'].map(facultades)

    # Eliminar la columna cod_facultad
    datos_combinados = datos_combinados.drop(columns=['cod_facultad'])

    # Reordenar columnas para que NombreFacultad sea la primera
    columnas = ['NombreFacultad'] + [col for col in datos_combinados.columns if col != 'NombreFacultad']
    datos_combinados = datos_combinados[columnas]

    # Guardar como CSV
    datos_combinados.to_csv(ruta_csv_salida, index=False, encoding='utf-8-sig')

# Ruta del archivo Excel y del archivo CSV de salida
ruta_excel = "datos.xlsx"  # Cambiar por la ruta real del archivo Excel
ruta_csv_salida = "datos_combinados.csv"

# Procesar el archivo y crear el CSV
procesar_excel_y_crear_csv(ruta_excel, ruta_csv_salida)

print(f"Archivo CSV generado correctamente en {ruta_csv_salida}")
