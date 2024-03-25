# Descripcion: Este script se encarga de reconocer los nombres de las personas en las fotos indicadas en la carpeta "fotos"

# importo librerias necesarias
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pixels import intensidad_pixels # Importo la funcion intensidad_pixels del archivo pixels.py creado anteriormente


#********************************************************************************************************************
#                     DEFINICION DE DIRECTORIOS Y ARCHIVOS
#********************************************************************************************************************
# Define the folder path
# Get the current working directory
current_directory = os.getcwd()
folder_name = "fotos"

# current_directory
# os.chdir(os.path.join(current_directory, "NuestrasCaras"))

folder_path = os.path.join(os.getcwd(), folder_name)

#********************************************************************************************************************
#                    OBTENCION DE LOS VALORES DE LOS PIXELES DE LAS FOTOS MEDIANTE PIXELS.PY
#********************************************************************************************************************
intensidad_pixels(folder_path)
# Toda la info se guarda en greyscale_values.csv

# Load the greyscale values from the CSV file
data_fotos = pd.read_csv('greyscale_values.csv')

# Get the file names
file_names = data_fotos.iloc[:, 0]

# Tomo los nombres de cada persona
people_names = [name.split("-")[0] for name in file_names]

#********************************************************************************************************************
#                    PCA
#********************************************************************************************************************

# Get the greyscale values
greyscale_values = data_fotos.iloc[:, 1:].values

# Standardize the greyscale values
scaler = StandardScaler()
greyscale_values_standardized = scaler.fit_transform(greyscale_values)

# Create a PCA object (tomo 20 CP)
#pca = PCA() # asi se pueden observar todas las CP
pca = PCA(n_components=20)

# Fit the PCA object
principal_components = pca.fit_transform(greyscale_values_standardized)

# Get the explained variance ratios
explained_variance_ratios = pca.explained_variance_ratio_

# Create a bar plot
plt.bar(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, alpha=0.5, align='center')
plt.xticks(range(1, len(explained_variance_ratios) + 1))
plt.xlabel('Componente Principal')
plt.ylabel('Proporción de Varianza Explicada')
plt.title('Proporción de Varianza Explicada por Componente Principal')
plt.show()

# Create a DataFrame to store the principal components
principal_components_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
principal_components_df

# Add the people names to the DataFrame
principal_components_df["People"] = people_names

# Save the DataFrame to a CSV file
principal_components_df.to_csv('componentes_principales.csv', index=False)

#********************************************************************************************************************
#                   CARA PROMEDIO CREADA CON EL PROMEDIO DE LOS VALORES GRISES DE LAS FOTOS
#********************************************************************************************************************
# Calculate the average greyscale values
average_greyscale_values = np.mean(greyscale_values, axis=0)

# Reshape the average greyscale values to match the image dimensions
average_greyscale_values = average_greyscale_values.reshape((30, 30))

# Convert the average greyscale values to uint8 data type
average_greyscale_values = average_greyscale_values.astype(np.uint8)

# Create an image from the average greyscale values
average_image = Image.fromarray(average_greyscale_values, mode='L')

# Display the average face
average_image.show()