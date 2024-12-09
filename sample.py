import pandas as pd

# Ruta del archivo CSV original
input_file = "./dataset/data/NF-UQ-NIDS-v2.csv"
# Ruta para guardar el archivo de muestra
output_file = "sample2.csv"

# Cargar el CSV en un DataFrame
data = pd.read_csv(input_file)

# Seleccionar las 10 primeras filas del DataFrame filtrado
sampled_data = data.head(50)

# Guardar la muestra en un nuevo archivo CSV
sampled_data.to_csv(output_file, index=False)

print(f"Archivo de muestra guardado en: {output_file}")
