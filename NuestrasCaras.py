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
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mode
from sklearn.decomposition import KernelPCA
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

# Create a PCA object (tomo 60 CP)
cant_componentes = 60

#pca = PCA() # asi se pueden observar todas las CP
pca = PCA(n_components=cant_componentes)

# Fit the PCA object
principal_components = pca.fit_transform(greyscale_values_standardized)

# peso de cada componente
peso_componentes = pca.components_

# Get the explained variance ratios
explained_variance_ratios = pca.explained_variance_ratio_

# Create a bar plot
#plt.bar(range(1, len(explained_variance_ratios) + 1), explained_variance_ratios, alpha=0.5, align='center')
#plt.xticks(range(1, len(explained_variance_ratios) + 1))
#plt.xlabel('Componente Principal')
#plt.ylabel('Proporción de Varianza Explicada')
#plt.title('Proporción de Varianza Explicada por Componente Principal')
#plt.show()

# Create a DataFrame to store the principal components
principal_components_df = pd.DataFrame(principal_components, columns = [f"PC{i}" for i in range(1, cant_componentes + 1)])
# imprime los componentes principales
#principal_components_df

# Add the people names to the DataFrame
principal_components_df["Persona"] = people_names

# Save the DataFrame to a CSV file
principal_components_df.to_csv('componentes_principales.csv', index=False)


#********************************************************************************************************************
#                   RECOMPOSICION DE IMAGENES
#********************************************************************************************************************

# Para desescalar los valores de los pixeles y volver a la escala original
original_mean = greyscale_values.mean()
original_std = greyscale_values.std()
scaler.mean_ = original_mean
scaler.scale_ = original_std

# Imagen original (elegi la nro 16)
plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
plt.imshow(greyscale_values[16].reshape(30, 30),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.title('900 componentes', fontsize = 20)
principal_components[16]

# Use the scaler's inverse_transform method to reverse the scaling
approximation = pca.inverse_transform(principal_components)
approximation_original_scale = scaler.inverse_transform(approximation)

plt.subplot(1, 2, 2)
plt.imshow(approximation_original_scale[16].reshape(30, 30),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.title('60 componentes', fontsize = 20)
plt.show()




#********************************************************************************************************************
#                   GRAFICOS DE LOS COMPONENTES PRINCIPALES
#********************************************************************************************************************
# Creo un plot de las primeras 20 componentes principales
primeras_componentes = pca.components_[:20]
# Crear una figura con 4 filas y 5 columnas para mostrar las primeras 20 componentes
fig, axs = plt.subplots(4, 5, figsize=(15, 12))
# Iterar sobre las primeras 20 componentes y mostrar cada una en un subplot
for i, componente in enumerate(primeras_componentes):
    ax = axs[i // 5, i % 5]
    ax.imshow(componente.reshape(30, 30), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Componente {i+1}')
# Ajustar espaciado entre subplots
plt.tight_layout()
plt.show()

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
#                   AGRUPACION CLUSTER DE LAS PERSONAS (agglomerative clustering, k-means, DBscan)
#********************************************************************************************************************
# 18 clusters ya que somos 18 personas
num_clusters = 18

#************* TECNICA 1: AGGLOMERATIVE CLUSTERING *******************
# Perform agglomerative clustering
agg_cluster = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')  # You can adjust the number of clusters and linkage method
cluster = agg_cluster.fit_predict(principal_components)

# Add the cluster labels to the DataFrame
principal_components_df["Cluster"] = cluster

# Create a DataFrame to store the photo number, person name, and cluster number
df_agg_clustering = pd.DataFrame({'Numero de Foto': range(1, len(file_names) + 1), 'Nombre de Persona': people_names, 'Numero de Cluster': cluster})

# Print the DataFrame
print(df_agg_clustering)

#************* TECNICA 2: K-MEANS CLUSTERING *******************
# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(principal_components)

# Add the cluster labels to the DataFrame
principal_components_df["Cluster"] = clusters

# Create a DataFrame to store the photo number, person name, and cluster number
df_kmeans = pd.DataFrame({'Numero de Foto': range(1, len(file_names) + 1), 'Nombre de Persona': people_names, 'Numero de Cluster': cluster})
print(df_kmeans)


#************* TECNICA 3: DBSCAN CLUSTERING *******************
# Perform K-means clustering
dbscan = DBSCAN(n_clusters=num_clusters)
clusters = DBSCAN.fit_predict(principal_components)

# Add the cluster labels to the DataFrame
principal_components_df["Cluster"] = clusters

# Create a DataFrame to store the photo number, person name, and cluster number
df_dbscan = pd.DataFrame({'Numero de Foto': range(1, len(file_names) + 1), 'Nombre de Persona': people_names, 'Numero de Cluster': cluster})
print(df_dbscan)

#*************************************************************
# Las tres tecnicas funcionan, hay que probar cual nos gusta mas (CREO que agglomerative funciona mejor)
#*************************************************************


#********************************************************************************************************************
#                   VISUALIZACION DE LOS CLUSTERS
#********************************************************************************************************************
# Visualize the clusters
plt.figure(figsize=(10, 10))
sns.scatterplot(x="PC1", y="PC2", data=principal_components_df, hue="Cluster", s=100)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Agrupación de las Fotos")
plt.show()







# Visualize the clusters
#plt.figure(figsize=(10, 10))
#sns.scatterplot(x="PC1", y="PC2", data=principal_components_df, hue="Cluster", s=100)
#plt.xlabel("Componente Principal 1")
#plt.ylabel("Componente Principal 2")
#plt.title("Agrupación de las Fotos")
#plt.show()




# ******** Ahora quiero saber cuantas fotos hay de cada persona en cada cluster
# y calcular el porcentaje de fotos de la persona que mas hay en ese cluster

# Count the number of photos for each person in each cluster
cluster_counts = principal_components_df.groupby(['Cluster', 'Persona']).size().reset_index(name='Count')
cluster_counts

# Cantidad de fotos de la persona que mas fotos tiene en cada cluster
# Si este valor es muy bajo, el cluster no tiene sentido
amount_most_pictures = cluster_counts.groupby('Cluster')['Count'].apply(lambda x: round(x.max(), 0))
amount_most_pictures

# Calculate the percentage of photos that the person with the most pictures has in each cluster
percentage_most_pictures = cluster_counts.groupby('Cluster')['Count'].apply(lambda x: round(x.max() / x.sum() * 100, 1))

# Get the person with the most pictures in each cluster
person_with_most_pictures = cluster_counts.loc[cluster_counts.groupby('Cluster')['Count'].idxmax(), 'Persona']

# Combine the percentage and person information into a DataFrame
df_clusters_por_persona = pd.DataFrame({'Cluster': percentage_most_pictures.index, 'Percentage': percentage_most_pictures.values,'Amount': amount_most_pictures, 'Person': person_with_most_pictures.values})
df_clusters_por_persona #Este df muestra el porcentaje de fotos de la persona que mas fotos tiene en cada cluster

#********************************************************************************************************************





#********************************************************************************************************************
#                   IDENTIFICACION DE LAS PERSONAS EN LAS FOTOS
#********************************************************************************************************************

folder_name_a_identificar = "fotos-a-identificar"

folder_path_a_identificar = os.path.join(os.getcwd(), folder_name_a_identificar)

# Load the photos to be identified
data_fotos_identificar = intensidad_pixels(folder_path_a_identificar)

# Get the greyscale values of the photos to be identified
greyscale_values_identificar = data_fotos_identificar.iloc[:, 1:].values

# Standardize the greyscale values of the photos to be identified
greyscale_values_identificar_standardized = scaler.transform(greyscale_values_identificar)






# Crear un objeto KernelPCA con kernel gaussiano y 200 componentes
kpca = KernelPCA(n_components=200, kernel='rbf')

# Ajustar y transformar tus fotos
fotos_reducidas = kpca.fit_transform(greyscale_values_identificar_standardized)

# Calcular la distancia de la foto a cada centroide de cluster
distancias_a_centroides = euclidean_distances(fotos_reducidas, kmeans.cluster_centers_)

# Encontrar el índice del cluster más cercano
indice_cluster_mas_cercano = np.argmin(distancias_a_centroides)

# Encontrar las personas asociadas a ese cluster
personas_en_cluster = people_names[clusters == indice_cluster_mas_cercano]

# Determinar la persona más común en ese cluster
persona_mas_comun = mode(personas_en_cluster).mode[0]