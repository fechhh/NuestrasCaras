# Descripcion: Este script se encarga de reconocer los nombres de las personas en las fotos indicadas en la carpeta "fotos"
#********************************************************************************************************************
# V.2:
# - Limpieza del codigo
# - Incorporación de TensorFlow para la creación de una red neuronal que permita identificar las personas en las fotos
#********************************************************************************************************************

# importo librerias necesarias
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mode
from sklearn.decomposition import KernelPCA

from pixels import intensidad_pixels # Importo la funcion intensidad_pixels del archivo pixels.py
from photo_30x30 import cortar_imagenes # Importo la funcion cortar_imagenes del archivo photo_30x30.py


#********************************************************************************************************************
#                     DEFINICION DE DIRECTORIOS Y ARCHIVOS
#********************************************************************************************************************
# Define the folder path
# Get the current working directory
current_directory = os.getcwd()
folder_name = "fotos_crudas"

# current_directory
# os.chdir(os.path.join(current_directory, "NuestrasCaras"))

folder_path = os.path.join(os.getcwd(), folder_name)

#********************************************************************************************************************
#                    CORTE DE LAS FOTOS Y CAMBIO DE ESCALA DE GRISES
#********************************************************************************************************************
# Cortar las fotos y cambiar a escala de grises
cortar_imagenes(folder_path, os.path.join(os.getcwd(), "fotos_output"))

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

# Crear funcion para la cara promedio
def cara_promedio(greyscale_values):
    # Calcular el promedio de los valores de los pixeles
    average_greyscale_values = np.mean(greyscale_values, axis=0)
    # Reshape the average greyscale values to match the image dimensions
    average_greyscale_values = average_greyscale_values.reshape((30, 30))
    # Convert the average greyscale values to uint8 data type
    average_greyscale_values = average_greyscale_values.astype(np.uint8)
    # Create an image from the average greyscale values
    average_image = Image.fromarray(average_greyscale_values, mode='L')
    return average_image

average_image = cara_promedio(greyscale_values)

# Display the average face
#average_image.show()

#********************************************************************************************************************
#                    PCA
#********************************************************************************************************************

# Crear funcion para PCA
def pca_caras(n_componentes, greyscale_values):
    # Standardize the greyscale values
    scaler = StandardScaler()
    greyscale_values_standardized = scaler.fit_transform(greyscale_values)
    # Create a PCA object
    pca = PCA(n_componentes)
    # Fit the PCA object
    principal_components = pca.fit_transform(greyscale_values_standardized)
    # Get the explained variance ratios
    explained_variance_ratios = pca.explained_variance_ratio_
    # Peso de cada componente
    peso_componentes = pca.components_
    return pca, principal_components, peso_componentes


# Create a PCA object (tomo 60 CP)
cant_componentes = 60

# Corro la funcion
pca, principal_components, peso_componentes = pca_caras(cant_componentes, greyscale_values)



#********************************************************************************************************************
#                   RECOMPOSICION DE IMAGENES
#********************************************************************************************************************

# Para desescalar los valores de los pixeles y volver a la escala original
scaler = StandardScaler()
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

# Imagen aproximada recreada a traves de las 60 CP
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

# Crear un plot de las primeras componentes principales

def crea_eigenfaces(n_componentes, greyscale_values):
    # Crear un objeto PCA con el número de componentes especificado
    pca = PCA(n_componentes)
    # Ajustar y transformar los valores de los pixeles
    principal_components = pca.fit_transform(greyscale_values)
    # Obtener los pesos de los componentes
    peso_componentes = pca.components_
    # Crear un plot de las imágenes con las primeras 20 componentes principales 
    primeras_componentes = peso_componentes[:n_componentes]
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

# Crear un plot de las primeras 20 componentes principales
crea_eigenfaces(20, greyscale_values)

    


# Create a pairplot of the principal components (tarda bastante)

# Create a DataFrame to store the principal components
principal_components_df = pd.DataFrame(principal_components, columns = [f"PC{i}" for i in range(1, cant_componentes + 1)])
# Add the people names to the DataFrame
principal_components_df["Persona"] = people_names
#sns.pairplot(principal_components_df, hue="Persona")
#plt.show()

# Create a scatter plot of the first two principal components
#plt.figure(figsize=(10, 10))
#sns.scatterplot(x="PC1", y="PC2", data=principal_components_df, hue="Persona", s=100)
#plt.xlabel("Componente Principal 1")
#plt.ylabel("Componente Principal 2")
#plt.title("Componentes Principales 1 y 2")
#plt.show()



#********************************************************************************************************************
#                   ARMADO DE MATRICES PARA CORRER LA RED NEURONAL
#********************************************************************************************************************

# Crear un objeto OneHotEncoder
encoder = OneHotEncoder()

# Convertir las etiquetas de las personas en una matriz dispersa
people_names_array = np.array(people_names).reshape(-1, 1)
people_names_sparse = encoder.fit_transform(people_names_array)
people_names_sparse

# Convertir la matriz dispersa a una matriz densa
people_names_dense = people_names_sparse.toarray()

# Convertir la matriz densa a un DataFrame
people_names_df = pd.DataFrame(people_names_dense, columns=encoder.categories_[0])
people_names_df

# Concatenar el DataFrame de las personas con el DataFrame de los componentes principales
principal_components_people_df = pd.concat([pd.DataFrame(principal_components), people_names_df], axis=1)
principal_components_people_df

#********************************************************************************************************************
#                   RED NEURONAL
#********************************************************************************************************************

# Define the number of input features
num_features = principal_components.shape[1]
num_features

# Define the number of output classes
num_classes = len(encoder.categories_[0])

# Create a neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(principal_components, people_names_dense, epochs=30, batch_size=32)

# Model evaluation
loss, accuracy = model.evaluate(principal_components, people_names_dense)
print(f"Loss: {loss}, Accuracy: {accuracy}")


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

# PCA of images to predict
principal_components_identificar = pca.transform(greyscale_values_identificar_standardized)

# Creamos las predicciones
predictions = model.predict(principal_components_identificar)
# Obtenemos la persona con el maximo de prediccion
predicted_classes = np.argmax(predictions, axis=1)
# Pasamos la prediccion a nombre
predicted_labels = encoder.categories_[0][predicted_classes]
# Obtenemos la probabilidad de exito
predicted_probabilities = np.round(np.max(predictions, axis=1), 2)

for i, prediccion, probabilidad in zip(range(len(predicted_labels)),predicted_labels, predicted_probabilities):
    print(f"Predicción {i+1}: {prediccion}, Probabilidad: {probabilidad}")


#********************************************************************************************************************