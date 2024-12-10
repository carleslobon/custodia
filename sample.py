import pandas as pd
import numpy as np

# Ruta del archivo CSV original
input_file = "./dataset/data/NF-UQ-NIDS-v2.csv"
# Ruta para guardar el archivo de muestra
output_file = "./samples/sample2.csv"

# Cargar el CSV en un DataFrame
data = pd.read_csv(input_file, nrows=150)

for col in data.select_dtypes(include=[np.number]).columns:
    upper_limit = data[col].quantile(0.99)
    lower_limit = data[col].quantile(0.01)
    data = data[(data[col] <= upper_limit) & (data[col] >= lower_limit)]

# Guardar la muestra en un nuevo archivo CSV
data.to_csv(output_file, index=False)

print(f"Archivo de muestra guardado en: {output_file}")
