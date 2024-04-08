## Nuestras Caras

# Objetivo
Diseñar un script que reciba imagenes y nombres de personas, las agrupe acorde a cada persona y luego pueda identificar nombres al brindarle nuevas imágenes.

# Funcionamiento
1) NuestrasCaras.py es el archivo principal que va a llamar a otros scripts.
2) pixels.py recibe las imagenes, las convierte a escala de grises y las transforma en 30x30 pixels (900 en total).
3) Crea un archivo llamado greyscale_values.csv con la información del nombre de la persona y la intensidad de cada pixel (0-255).
4) photo_30x30.ipynb . Nootebook que a partir de fotos crudas de caras tomadas con cámara crea sobre la carpeta output una extracción 30x30 únicamente del rostro y en escala de grises.
5) 