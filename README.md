# Nuestras Caras

## Grupo 1
* Grijalba, Federico
* Richard, Federico
* Ruiz, Lautaro
* Szekieta, Paola

## Objetivo
Diseñar un script que reciba imagenes y nombres de personas, las agrupe acorde a cada persona y luego pueda identificar nombres al brindarle nuevas imágenes.

## Dependencias/Librerias necesarias (como instalar ambiente)
Para realizar la ejecución de las notebooks es necesario realizar la intalacion de las dependencias del proyecto.

Las mismas se encentran en el archivo [requirements.txt](./requirements.txt).

Para su instalación mediante pip, se debe ejecutar desde la terminal:
```sh
pip install -r /path/to/requirements.txt
```

Recordar tener activado el ambiente del proyecto. El mismo puede ser definido con virtualenv.

Es opcional la instalación de 'tensorflow'.

Si se tiene instalado conda, se puede crear el ambiente mediante el archivo [environment.yml](./environment.yml). Para esto ejecutar desde la terminal:
```sh
conda env create -f environment.yml
```

## Entrenamiento Modelo

El entrenamiento del modelo para el reconocimiento de las caras es realizado mediante la notebook `01_entrenar_modelo\01_entrenar_modelo.ipynb`. 

## Evaluación nuevas imagenes

Mediante la notebook `02_probar_nuevas_fotos\02_probar_nuevas_fotos.ipynb`.

Pasos para la evaluación de nuevas imágenes:
1. Colocar las nuevas imágenes con formato "jpg" o "jpeg" en la carpeta "**02_probar_nuevas_fotos\fotos_prueba**".

    Es importante que las mismas tengan el nombre de la persona en minuscula y un número en nombre del archivo.

    Por ejemplo: "paola1.jpeg", "paola2.jpeg" o "lautaro1.jpeg", "lautaro2.jpeg".

    En caso de tener personas con el mismo nombre, colocar el nombre en minuscula, seguido de la primera letra del apellido en mayuscula.

    Por ejemplo: "federicoG1.jpeg", "federicoG2.jpeg" o "federicoR1.jpeg", "federicoR2.jpeg".

2. Ejecutar la notebook `02_probar_nuevas_fotos\02_probar_nuevas_fotos.ipynb`.

    Al final de la misma se verificará el porcentaje de aciertos del modelo, indicando los nombres de las personas reconocidas y no reconocidas.

## EXTRA: Entrenamiento y evaluación con tensorflow
Se se armo la notebook `03_pruebas_entrenamiento_tensorflow\03_entrenar_modelo_y_probar_con_tensorflow.ipynb` mediante la cual se plantean varias pruebas con tensorflow.

## Mejoras

## FALTA:
correr algun modelito pasable (demora un rato largo por eso no lo hice aun pero el finde se puede hacer)

despues pasarlo a main y borrar todas las ramas que queden dando vuelta

gitignorar las imagenes en el repo