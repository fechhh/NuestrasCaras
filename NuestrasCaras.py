# Descripcion: Este script se encarga de reconocer los nombres de las personas en las fotos indicadas en la carpeta "fotos"

# importo librerias necesarias
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import pandas as pd
from pixels import intensidad_pixels # Importo la funcion intensidad_pixels del archivo pixels.py creado anteriormente

# Define the folder path
# Get the current working directory
current_directory = os.getcwd()
folder_name = "fotos"

# current_directory
# os.chdir(os.path.join(current_directory, "NuestrasCaras"))

folder_path = os.path.join(os.getcwd(), folder_name)

intensidad_pixels(folder_path)