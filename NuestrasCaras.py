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
from sklearn.cluster import KMeans
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
# Guardo la info de las fotos en un dataframe
data_fotos = intensidad_pixels(folder_path)

# Get the file names
file_names = data_fotos.iloc[:, 0]

# Tomo los nombres de cada persona
people_names = [name.split("-")[0] for name in file_names]

# Get the greyscale values
greyscale_values = data_fotos.iloc[:, 1:].values

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
#average_image.show()

#********************************************************************************************************************
#                    PCA
#********************************************************************************************************************

# Standardize the greyscale values
scaler = StandardScaler()
greyscale_values_standardized = scaler.fit_transform(greyscale_values)

# Create a PCA object (tomo 20 CP)
cant_componentes = 50

#pca = PCA() # asi se pueden observar todas las CP
pca = PCA(n_components=cant_componentes)

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
#plt.show()

# Create a DataFrame to store the principal components
principal_components_df = pd.DataFrame(principal_components)
# imprime los componentes principales
#principal_components_df

# Add the people names to the DataFrame
principal_components_df["Persona"] = people_names

# Save the DataFrame to a CSV file
principal_components_df.to_csv('componentes_principales.csv', index=False)

#********************************************************************************************************************
#                   GRAFICOS DE LOS COMPONENTES PRINCIPALES (tarda en ejecutar)
#********************************************************************************************************************
# Create a pairplot of the principal components (tarda bastante)
#sns.pairplot(principal_components_df, hue="Persona")
#plt.show()

# Create a scatter plot of the first two principal components
#plt.figure(figsize=(10, 10))
#sns.scatterplot(x="PC1", y="PC2", data=principal_components_df, hue="Persona", s=100)
#plt.xlabel("Componente Principal 1")
#plt.ylabel("Componente Principal 2")
#plt.title("Componentes Principales 1 y 2")
#plt.show()

# Create a scatter plot of the first and third principal components
#plt.figure(figsize=(10, 10))
#sns.scatterplot(x="PC1", y="PC3", data=principal_components_df, hue="Persona", s=100)
#plt.xlabel("Componente Principal 1")
#plt.ylabel("Componente Principal 3")
#plt.title("Componentes Principales 1 y 3")
#plt.show()

#********************************************************************************************************************
#                   AGRUPACION CLUSTER DE LAS PERSONAS
#********************************************************************************************************************
# 18 clusters ya que somos 18 personas
num_clusters = 18

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(principal_components)

# Add the cluster labels to the DataFrame
principal_components_df["Cluster"] = clusters

# Visualize the clusters
#plt.figure(figsize=(10, 10))
#sns.scatterplot(x="PC1", y="PC2", data=principal_components_df, hue="Cluster", s=100)
#plt.xlabel("Componente Principal 1")
#plt.ylabel("Componente Principal 2")
#plt.title("Agrupación de las Fotos")
#plt.show()

# Create a DataFrame to store the photo number, person name, and cluster number
df = pd.DataFrame({'Numero de Foto': range(1, len(file_names) + 1),
                   'Nombre de Persona': people_names,
                   'Numero de Cluster': clusters})

# Print the DataFrame
print(df)

# Count the number of photos for each person in each cluster
cluster_counts = principal_components_df.groupby(['Cluster', 'Persona']).size().reset_index(name='Count')

# Find the person with the maximum count in each cluster
max_counts = cluster_counts.groupby('Cluster')['Count'].idxmax()

# Get the corresponding person names for each cluster
cluster_names = cluster_counts.loc[max_counts, ['Cluster', 'Persona']]

# Calculate the probability of each person in each cluster
cluster_probabilities = cluster_counts.groupby('Cluster').apply(lambda x: x['Count'] / x['Count'].sum()).reset_index(name='Probability')

# Merge the cluster names and probabilities
cluster_names = cluster_names.merge(cluster_probabilities, on='Cluster')

# Sort the clusters by probability in descending order
cluster_names = cluster_names.sort_values(by='Probability', ascending=False)

# Print the cluster names
print(cluster_names)

#********************************************************************************************************************
#                   DADA UNA FOTO IDENTIFICAR PERSONA 
#********************************************************************************************************************
folder_name = "fotos-a-identificar"

# current_directory
# os.chdir(os.path.join(current_directory, "NuestrasCaras"))

folder_path = os.path.join(os.getcwd(), folder_name)

# Load the photos to be identified
data_fotos_identificar = intensidad_pixels(folder_path)

# Get the greyscale values of the photos to be identified
greyscale_values_identificar = data_fotos_identificar.iloc[:, 1:].values

# Standardize the greyscale values of the photos to be identified
greyscale_values_identificar_standardized = scaler.transform(greyscale_values_identificar)

