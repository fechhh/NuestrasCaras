## defino funcion para cortar caras de una imagen
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import face_recognition
from PIL import Image
import re

# Definir dimension de las imagenes
DIM_FOTOS = 30
NUEVAS_IMAGEMES_GENERAR = 30


def preparar_imagenes_entrenamiento(
    input, output, prefix="gen_aut", cantidad_imagenes=NUEVAS_IMAGEMES_GENERAR
):
    """Preparar imagenes para el entrenamiento de los modelos"""
    # 1° Recorta las imagenes
    print("Recortando imagenes para el entrenamiento del modelo...")
    cortar_imagenes(input, output)

    # 2° Genera imagenes
    print(
        f"Generando {cantidad_imagenes} imagenes nuevas para cada imagen disponible..."
    )
    gen_new_image(output, prefix, cantidad_imagenes)

    print("Imagenes listas para el entrenamiento del modelo!")


def preparar_imagenes_testeo(input, output):
    """Preparar imagenes para el testeo de los modelos"""
    # 1° Recorta las imagenes
    print("Recortando imagenes para el testeo del modelo...")
    cortar_imagenes(input, output)
    print("Imagenes listas para el testeo del modelo!")


def eliminar_numeros(texto):
    """Eliminar numeros de un string"""
    return re.sub(r"\d+", "", texto)


def cortar_imagenes(input_dir, output_dir, dim=DIM_FOTOS):
    """Cortar las caras de las personas identificadas en las imágenes"""
    # Cargamos el detector de rostros de la libreria face_recognition
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    # Para cada archivo del directorio de imput comenzamos el loop
    for filename in os.listdir(input_dir):
        i += 1
        # Cargamos la imagen
        input_path = os.path.join(input_dir, filename)
        img = cv2.imread(input_path)

        # Detectamos el rostro en la imagen
        face_locations = face_recognition.face_locations(img)

        # Cortamos y cambiamos a escala de grises
        for top, right, bottom, left in face_locations:
            face = img[top:bottom, left:right]
            face = cv2.resize(face, (dim, dim))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Guardamos el proceso en la carpeta de salida
            output_path = os.path.join(output_dir, f"{filename}")
            cv2.imwrite(output_path, face)


def gen_new_image(folder_path, prefix, cantidad_imagenes):
    """Generar imagenes aleatorias a partir de imagenes existentes"""

    # obtiene nombre de los archivos con las imagenes
    file_names = os.listdir(folder_path)

    # itera sobre los archivos
    for file_name in file_names:
        # arma ruta a la imagen
        file_path = os.path.join(folder_path, file_name)

        # carga la imagen
        img = image.load_img(file_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # crea un generador de datos con aumentos
        datagen = ImageDataGenerator(
            rotation_range=3,
            width_shift_range=0.025,
            height_shift_range=0.025,
            shear_range=0.025,
            zoom_range=0.025,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        # separa nombre archivo y extension (a usar en el nombre de la nueva imagen generada)
        name, ext = os.path.splitext(file_name)

        # inicializa el bucle para las 'cantidad_imagenes' a generar
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            # define nombre de la nueva imagen generada segun prefijo
            new_filename = f"{name}_{prefix}_{i}{ext}"

            # guarda la imagen aumentada
            new_file_path = os.path.join(folder_path, new_filename)
            img_augmented = image.array_to_img(batch[0])
            img_augmented.save(new_file_path)
            i += 1
            if i >= cantidad_imagenes:
                break


# ejemplo de uso
# cortar_imagenes(input_dir="fotos_probamos_distintas_opciones/input", output_dir="fotos_probamos_distintas_opciones/output")
# gen_new_image(folder_path="fotos_probamos_distintas_opciones/output", prefix="gen_aut", cantidad_imagenes=10)
