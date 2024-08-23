# backpropagation, just one hidden layer
# lo hago con  matrices de pesos
# puedo tener tantos inputs como quiera
# puedo tener tantas neuronas ocultas como quiera
# puedo tener tanas neuronas de salida como quiera
# fuera de este codigo esta la decision que tomo segun el valor de salida de cada neurona de salida

import math
import numpy as np
import pandas as pd


def func_eval(fname, x):
    match fname:
        case "purelin":
            y = x
        case "logsig":
            y = 1.0 / (1.0 + math.exp(-x))
        case "tansig":
            y = 2.0 / (1.0 + math.exp(-2.0 * x)) - 1.0
    return y


func_eval_vec = np.vectorize(func_eval)


def deriv_eval(fname, y):  # atencion que y es la entrada y=f( x )
    match fname:
        case "purelin":
            d = 1.0
        case "logsig":
            d = y * (1.0 - y)
        case "tansig":
            d = 1.0 - y * y
    return d


deriv_eval_vec = np.vectorize(deriv_eval)


def backpropagation_2_layers(
    X,
    Y,
    hidden_size1,
    hidden_size2,
    epoch_limit=200,
    learning_rate=0.8,
    Error_umbral=0.00000000001,
) -> dict:
    """
    Aplicar metodo de Backpropagation para entrenar una red neuronal con 2 capas ocultas

    Parámetros:
    X: np.array
        Datos de entrada
    Y: np.array
        Datos de salida
    hidden_size1: int
        Cantidad de neuronas de la primera capa oculta
    hidden_size2: int
        Cantidad de neuronas de la segunda capa oculta
    epoch_limit: int
        Cantidad máxima de iteraciones
    learning_rate: float
        Tasa de aprendizaje
    Error_umbral: float
        Tolerancia del error

    Returns:
    dict
        Pesos obtenidos en el entrenamiento para cada capa
    """

    # define dimensiones
    filas_qty = len(X)  # Cantidad de registros
    input_size = X.shape[1]  # Capa entrada
    output_size = Y.shape[1]  # Capa salida

    # define funciones de activacion de cada capa
    hidden_FUNC_1 = "logsig"  # uso la logística
    hidden_FUNC_2 = "logsig"  # uso la logística
    output_FUNC = "logsig"  # uso la logística

    # Incializo las matrices de pesos aleatoriamente
    np.random.seed(
        1021
    )  # mi querida random seed para que las corridas sean reproducibles
    W1 = np.random.uniform(-0.5, 0.5, [hidden_size1, input_size])
    X01 = np.random.uniform(-0.5, 0.5, [hidden_size1, 1])
    W2 = np.random.uniform(-0.5, 0.5, [hidden_size2, hidden_size1])
    X02 = np.random.uniform(-0.5, 0.5, [hidden_size2, 1])
    W3 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size2])
    X03 = np.random.uniform(-0.5, 0.5, [output_size, 1])

    # Avanzo la red, forward
    hidden_estimulos1 = W1 @ X.T + X01
    hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
    hidden_estimulos2 = W2 @ hidden_salidas1 + X02
    hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
    output_estimulos = W3 @ hidden_salidas2 + X03
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    # Calculo el error promedio general de todos los X
    Error = np.mean((Y.T - output_salidas) ** 2)

    # Inicializo
    Error_last = 10  # lo debo poner algo dist a 0 la primera vez
    epoch = 0

    while math.fabs(Error_last - Error) > Error_umbral and (epoch < epoch_limit):
        epoch += 1
        Error_last = Error

        # Recorro siempre TODA la entrada
        for fila in range(filas_qty):
            # Propagar el x hacia adelante
            hidden_estimulos1 = W1 @ X[fila : fila + 1, :].T + X01
            hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
            hidden_estimulos2 = W2 @ hidden_salidas1 + X02
            hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
            output_estimulos = W3 @ hidden_salidas2 + X03
            output_salidas = func_eval_vec(output_FUNC, output_estimulos)

            # Calcular los errores en la capa hidden y la capa output
            ErrorSalida = Y[fila : fila + 1, :].T - output_salidas
            output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
            hidden_delta2 = deriv_eval_vec(hidden_FUNC_2, hidden_salidas2) * (
                W3.T @ output_delta
            )
            hidden_delta1 = deriv_eval_vec(hidden_FUNC_1, hidden_salidas1) * (
                W2.T @ hidden_delta2
            )

            # Corregir matrices de pesos, voy hacia atrás (backpropagation)
            W1 = W1 + learning_rate * (hidden_delta1 @ X[fila : fila + 1, :])
            X01 = X01 + learning_rate * hidden_delta1
            W2 = W2 + learning_rate * (hidden_delta2 @ hidden_salidas1.T)
            X02 = X02 + learning_rate * hidden_delta2
            W3 = W3 + learning_rate * (output_delta @ hidden_salidas2.T)
            X03 = X03 + learning_rate * output_delta

        # Avanzo la red, feed-forward
        hidden_estimulos1 = W1 @ X.T + X01
        hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
        hidden_estimulos2 = W2 @ hidden_salidas1 + X02
        hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
        output_estimulos = W3 @ hidden_salidas2 + X03
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)

        # Calcular el error promedio general de TODOS los X
        Error = np.mean((Y.T - output_salidas) ** 2)

        if epoch % 10 == 0:
            # muestra iteracion si es multiplo de 10
            print(f"Iteracion {epoch} - Error: {Error}")

    # Devuelve diccionario con los pesos y las funciones de activación por capa
    return {
        "hidden_weights1_func": [W1, X01, hidden_FUNC_1],
        "hidden_weights2_func": [W2, X02, hidden_FUNC_2],
        "output_weights_func": [W3, X03, output_FUNC],
    }


def get_predictions(x_new, weights_dict):

    # obtiene los pesos de la red entrenada y las funciones de activación
    W1 = weights_dict.get("hidden_weights1_func")[0]
    X01 = weights_dict.get("hidden_weights1_func")[1]
    hidden_FUNC_1 = weights_dict.get("hidden_weights1_func")[2]
    W2 = weights_dict.get("hidden_weights2_func")[0]
    X02 = weights_dict.get("hidden_weights2_func")[1]
    hidden_FUNC_2 = weights_dict.get("hidden_weights2_func")[2]
    W3 = weights_dict.get("output_weights_func")[0]
    X03 = weights_dict.get("output_weights_func")[1]
    output_FUNC = weights_dict.get("output_weights_func")[2]

    # predict (para clasificar un nuevo punto luego de ajustar la red!)
    hidden_estimulos1 = W1 @ x_new.T + X01
    hidden_salidas1 = func_eval_vec(hidden_FUNC_1, hidden_estimulos1)
    hidden_estimulos2 = W2 @ hidden_salidas1 + X02
    hidden_salidas2 = func_eval_vec(hidden_FUNC_2, hidden_estimulos2)
    output_estimulos = W3 @ hidden_salidas2 + X03
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    return output_salidas.T


def evaluate_predictions(x_new, y_labels, y_value, weights_dict):

    # obtiene las predicciones
    output_salidas = get_predictions(x_new, weights_dict)

    # arma dataset
    df_output_salidas = pd.DataFrame(output_salidas, columns=y_labels)
    df_output_salidas["nombre_real"] = y_value

    # Obtener el valor máximo y el nombre de la columna correspondiente para cada fila
    max_values = df_output_salidas.iloc[:, :-1].max(axis=1)
    max_columns = df_output_salidas.iloc[:, :-1].idxmax(axis=1)
    nombres_verdaderos = df_output_salidas.iloc[:, -1]

    # Construir el DataFrame con los resultados
    return pd.DataFrame(
        {
            "Valor máximo": max_values,
            "Nombre de la columna": max_columns,
            "Nombre verdadero": nombres_verdaderos,
        }
    )
