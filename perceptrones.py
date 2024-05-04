# backpropagation, just one hidden layer
# lo hago con  matrices de pesos
# puedo tener tantos inputs como quiera
# puedo tener tantas neuronas ocultas como quiera
# puedo tener tanas neuronas de salida como quiera
# fuera de este codigo esta la decision que tomo segun el valor de salida de cada neurona de salida

import math
import numpy as np

def func_eval(fname, x):
    match fname:
        case "purelin":
            y = x
        case "logsig":
            y = 1.0 / (1.0 + np.exp(-x))
        case "tansig":
            y = 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0
        case "relu":
            y = np.maximum(0, x)
    return y


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0))
    return exp_x / np.sum(exp_x, axis=0)

def deriv_eval(fname, y):  #atencion que y es la entrada y=f( x )
    match fname:
        case "purelin":
            d = 1.0
        case "logsig":
            d = y*(1.0-y)
        case "tansig":
            d = 1.0 - y*y
    return d


def backpropagation_1_capa(entrada, salida, n_neuronas, epoch_limit, Error_umbral, learning_rate):
    # Paso las listas a numpy
    X = np.array(entrada)
    Y = np.array(salida).reshape(len(X),1)
    
    # Vectorizar la función para poder pasarle un vector para los cálculos
    func_eval_vec = np.vectorize(func_eval)

    # Vectorizar la función para poder pasarle un vector para los cálculos
    deriv_eval_vec = np.vectorize(deriv_eval)

    filas_qty = len(X)
    input_size = X.shape[1]   # 2 entradas
    hidden_size = n_neuronas  # neuronas capa oculta
    output_size = Y.shape[1]  # 1 neurona

    # defino las funciones de activacion de cada capa
    hidden_FUNC = 'logsig'  # uso la logistica
    output_FUNC = 'tansig'  # uso la tangente hiperbolica


    # Incializo las matrices de pesos azarosamente
    # W1 son los pesos que van del input a la capa oculta
    # W2 son los pesos que van de la capa oculta a la capa de salida
    np.random.seed(1021) #mi querida random seed para que las corridas sean reproducibles
    W1 = np.random.uniform(-0.5, 0.5, [hidden_size, input_size])
    X01 = np.random.uniform(-0.5, 0.5, [hidden_size, 1] )
    W2 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size])
    X02 = np.random.uniform(-0.5, 0.5, [output_size, 1] )

    # Avanzo la red, forward
    # para TODOS los X al mismo tiempo ! 
    #  @ hace el producto de una matrix por un vector_columna
    hidden_estimulos = W1 @ X.T + X01
    hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
    output_estimulos = W2 @ hidden_salidas + X02
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    # calculo el error promedi general de TODOS los X
    Error= np.mean( (Y.T - output_salidas)**2 )


    # Inicializo
    Error_last = 10    # lo debo poner algo dist a 0 la primera vez
    epoch = 0

    while ( math.fabs(Error_last-Error)>Error_umbral and (epoch < epoch_limit)):
        epoch += 1
        Error_last = Error

        # recorro siempre TODA la entrada
        for fila in range(filas_qty): #para cada input x_sub_fila del vector X
            # propagar el x hacia adelante
            hidden_estimulos = W1 @ X[fila:fila+1, :].T + X01
            hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
            output_estimulos = W2 @ hidden_salidas + X02
            output_salidas = func_eval_vec(output_FUNC, output_estimulos)

            # calculo los errores en la capa hidden y la capa output
            ErrorSalida = Y[fila:fila+1,:].T - output_salidas
            # output_delta es un solo numero
            output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
            # hidden_delta es un vector columna
            hidden_delta = deriv_eval_vec(hidden_FUNC, hidden_salidas)*(W2.T @ output_delta)

            # ya tengo los errores que comete cada capa
            # corregir matrices de pesos, voy hacia atras
            # backpropagation
            W1 = W1 + learning_rate * (hidden_delta @ X[fila:fila+1, :] )
            X01 = X01 + learning_rate * hidden_delta
            W2 = W2 + learning_rate * (output_delta @ hidden_salidas.T)
            X02 = X02 + learning_rate * output_delta

        # ya recalcule las matrices de pesos
        # ahora avanzo la red, feed-forward
        hidden_estimulos = W1 @ X.T + X01
        hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
        output_estimulos = W2 @ hidden_salidas + X02
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)

        # calculo el error promedio general de TODOS los X
        Error= np.mean( (Y.T - output_salidas)**2 )
    
    return W1, X01, W2, X02, epoch, Error





def backpropagation_2_capas(entrada, salida, n_neuronas_capa1, n_neuronas_capa2, epoch_limit, Error_umbral, learning_rate):
    # Paso las listas a numpy
    X = np.array(entrada)
    Y = salida
    
    # Vectorizar la función para poder pasarle un vector para los cálculos
    func_eval_vec = np.vectorize(func_eval)

    # Vectorizar la función para poder pasarle un vector para los cálculos
    deriv_eval_vec = np.vectorize(deriv_eval)

    filas_qty = len(X)
    input_size = X.shape[1]   # 2 entradas
    hidden_size1 = n_neuronas_capa1  # neuronas capa oculta 1
    hidden_size2 = n_neuronas_capa2  # neuronas capa oculta 2
    output_size = Y.shape[1]  # 1 neurona

    # defino las funciones de activacion de cada capa
    hidden_FUNC1 = 'logsig'  # uso la logistica
    hidden_FUNC2 = 'logsig'  # uso la logistica
    output_FUNC = 'logsig'  # uso la tangente hiperbolica


    # Incializo las matrices de pesos azarosamente
    # W1 son los pesos que van del input a la capa oculta 1
    # W2 son los pesos que van de la capa oculta 1 a la capa oculta 2
    # W3 son los pesos que van de la capa oculta 2 a la capa de salida
    np.random.seed(1021) #mi querida random seed para que las corridas sean reproducibles
    W1 = np.random.uniform(-0.5, 0.5, [hidden_size1, input_size])
    X01 = np.random.uniform(-0.5, 0.5, [hidden_size1, 1] )
    W2 = np.random.uniform(-0.5, 0.5, [hidden_size2, hidden_size1])
    X02 = np.random.uniform(-0.5, 0.5, [hidden_size2, 1] )
    W3 = np.random.uniform(-0.5, 0.5, [output_size, hidden_size2])
    X03 = np.random.uniform(-0.5, 0.5, [output_size, 1] )
    
    # Avanzo la red, forward
    # para TODOS los X al mismo tiempo !
    hidden_estimulos1 = W1 @ X.T + X01
    hidden_salidas1 = func_eval_vec(hidden_FUNC1, hidden_estimulos1)
    hidden_estimulos2 = W2 @ hidden_salidas1 + X02
    hidden_salidas2 = func_eval_vec(hidden_FUNC2, hidden_estimulos2)
    output_estimulos = W3 @ hidden_salidas2 + X03
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)
    
    # calculo el error promedio general de TODOS los X
    Error= np.mean( (Y.T - output_salidas)**2 )
    print(f"Error inicial: {Error}")
    
    # Inicializo
    Error_last = 10    # lo debo poner algo dist a 0 la primera vez
    epoch = 0
    # epoch_limit, Error_umbral, learning_rate son parametros de ingreso a la funcion
    
    while ( math.fabs(Error_last-Error)>Error_umbral and (epoch < epoch_limit)):
        epoch += 1
        Error_last = Error
        
        # recorro siempre TODA la entrada
        for fila in range(filas_qty):
            # propagar el x hacia adelante
            hidden_estimulos1 = W1 @ X[fila:fila+1, :].T + X01
            hidden_salidas1 = func_eval_vec(hidden_FUNC1, hidden_estimulos1)
            hidden_estimulos2 = W2 @ hidden_salidas1 + X02
            hidden_salidas2 = func_eval_vec(hidden_FUNC2, hidden_estimulos2)
            output_estimulos = W3 @ hidden_salidas2 + X03
            output_salidas = func_eval_vec(output_FUNC, output_estimulos)
            
            # calculo los errores en la capa hidden y la capa output
            ErrorSalida = Y[fila:fila+1,:].T - output_salidas
            output_delta = ErrorSalida * deriv_eval_vec(output_FUNC, output_salidas)
            hidden_delta2 = deriv_eval_vec(hidden_FUNC2, hidden_salidas2)*(W3.T @ output_delta)
            hidden_delta1 = deriv_eval_vec(hidden_FUNC1, hidden_salidas1)*(W2.T @ hidden_delta2)
            
            # ya tengo los errores que comete cada capa
            # corregir matrices de pesos, voy hacia atras
            # backpropagation
            W1 = W1 + learning_rate * (hidden_delta1 @ X[fila:fila+1, :] )
            X01 = X01 + learning_rate * hidden_delta1
            W2 = W2 + learning_rate * (hidden_delta2 @ hidden_salidas1.T)
            X02 = X02 + learning_rate * hidden_delta2
            W3 = W3 + learning_rate * (output_delta @ hidden_salidas2.T)
            X03 = X03 + learning_rate * output_delta
            
        # ya recalcule las matrices de pesos
        # ahora avanzo la red, feed-forward
        hidden_estimulos1 = W1 @ X.T + X01
        hidden_salidas1 = func_eval_vec(hidden_FUNC1, hidden_estimulos1)
        hidden_estimulos2 = W2 @ hidden_salidas1 + X02
        hidden_salidas2 = func_eval_vec(hidden_FUNC2, hidden_estimulos2)
        output_estimulos = W3 @ hidden_salidas2 + X03
        output_salidas = func_eval_vec(output_FUNC, output_estimulos)
        
        # calculo el error promedio general de TODOS los X
        Error= np.mean( (Y.T - output_salidas)**2 )

    # Imprimo epoch y error
    print(f"Epoch: {epoch}, Error: {Error}")
        
    return W1, X01, W2, X02, W3, X03, Error, epoch



def predecir_clase_1_capa(pesos, entrada):
    # Desempaquetar los pesos
    W1, X01, W2, X02 = pesos
    
    entrada = np.array(entrada)

    # defino las funciones de activacion de cada capa
    hidden_FUNC = 'logsig'  # uso la logistica

    func_eval_vec = np.vectorize(func_eval)

    # Avanzar la red, forward
    hidden_estimulos = np.dot(W1, entrada.T) + X01
    hidden_salidas = func_eval_vec(hidden_FUNC, hidden_estimulos)
    output_estimulos = np.dot(W2, hidden_salidas) + X02
    output_salidas = softmax(output_estimulos)

    # Obtener la clase predicha
    clase_predicha = np.argmax(output_salidas)

    return output_salidas, clase_predicha



def predecir_clase_2_capas(pesos, entrada):
    # Desempaquetar los pesos
    W1, X01, W2, X02, W3, X03 = pesos

    entrada = np.array(entrada)

    # Definir las funciones de activación de cada capa
    hidden_FUNC1 = 'logsig'  # uso la logistica
    hidden_FUNC2 = 'logsig'  # uso la logistica
    output_FUNC = 'logsig'  # uso la logistica

    func_eval_vec = np.vectorize(func_eval)

    # Avanzar la red, forward
    hidden_estimulos1 = W1 @ entrada.T + X01
    hidden_salidas1 = func_eval_vec(hidden_FUNC1, hidden_estimulos1)
    hidden_estimulos2 = W2 @ hidden_salidas1 + X02
    hidden_salidas2 = func_eval_vec(hidden_FUNC2, hidden_estimulos2)
    output_estimulos = W3 @ hidden_salidas2 + X03
    output_salidas = func_eval_vec(output_FUNC, output_estimulos)

    # Obtener la clase predicha
    clase_predicha = np.argmax(output_salidas)

    return output_salidas, clase_predicha

