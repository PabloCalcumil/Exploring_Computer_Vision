import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from sklearn.model_selection import train_test_split
import boto3
from tensorflow.keras.utils import to_categorical

#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
def graphs_train_models(history, early, metric1 = 'accuracy', metric2 = 'AUC'):
    '''
    Funcion para formar graficos de evolucion de metricas y de funcion de perdida
    durante el entrenamiento de redes neuronales (Caso especifico Accuracy y AUC).
    
    Inputs:
        history : diccionario con valores de la loss y las metricas para cada
                  epoca de entrenamiento.
        early : numero de epocas de paciencia del metodo EarlyStopping.
        metric1 : string "accuracy" normalmente para ver evolucion de esta.
        metric2 : string "AUC" normalmente para ver evolucion de esta.
        
    Outputs:
        Solo grafica las evoluciones de las metricas y la loss de la red neuronal.
    '''
    #Largo total del entrenamiento (con EarleStopping no necesariamente se cumplen las epocas determinadas)
    len_histo = len(history.history['loss'])
    #Los graficos comenzaran desde la epoca 1 (sin esto comienzan desde la 0)
    range_epochs = list(range(1, len_histo + 1))
    #Figura de graficos
    fig, ax = plt.subplots(1, 3, figsize = (20, 5))
    #---------------------------- Graphs Loss ----------------------------#  
    sns.lineplot(x = range_epochs, y = history.history['loss'], label = 'Loss Train', color = 'blue', ax = ax[0])
    sns.lineplot(x = range_epochs, y = history.history['val_loss'], label = 'Loss Valid', color = 'orange', ax = ax[0])
    sns.scatterplot(x = range_epochs[4 : : 5], y = history.history['loss'][4 : : 5], color = 'blue', marker = 'o', s = 18, ax = ax[0])
    sns.scatterplot(x = range_epochs[4 : : 5], y = history.history['val_loss'][4 : : 5], color = 'orange', marker = 'o', s = 18, ax = ax[0])
    ax[0].axvline(x = len_histo - early, color = 'gray', linestyle = '--', label = 'Early Stopping')
    ax[0].set_xlabel('# Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc = 'upper right')
    #---------------------------- Graphs Metric1 ----------------------------#
    sns.lineplot(x = range_epochs, y = history.history[metric1], label = f'{metric1.capitalize()} Train', color = 'blue', ax = ax[1])
    sns.lineplot(x = range_epochs, y = history.history[f'val_{metric1}'], label = f'{metric1.capitalize()} Valid', color = 'orange', ax = ax[1])
    sns.scatterplot(x = range_epochs[4 : : 5], y = history.history[metric1][4 : : 5], color = 'blue', marker = 'o', s = 15, ax = ax[1])
    sns.scatterplot(x = range_epochs[4 : : 5], y = history.history[f'val_{metric1}'][4 : : 5], color = 'orange', marker = 'o', s = 15, ax = ax[1])
    ax[1].axvline(x = len_histo - early, color = 'gray', linestyle = '--', label = 'Early Stopping')
    ax[1].set_xlabel('# Epochs')
    ax[1].set_ylabel(f'{metric1.capitalize()}')
    ax[1].legend(loc = 'lower right')
    #---------------------------- Graphs Metric1 ----------------------------#
    sns.lineplot(x = range_epochs, y = history.history[metric2.lower()], label = f'{metric2} Train', color = 'blue', ax = ax[2])
    sns.lineplot(x = range_epochs, y = history.history[f'val_{metric2.lower()}'], label = f'{metric2} Valid', color = 'orange', ax = ax[2])
    sns.scatterplot(x = range_epochs[4 : : 5], y = history.history[metric2.lower()][4 : : 5], color = 'blue', marker = 'o', s = 15, ax = ax[2])
    sns.scatterplot(x = range_epochs[4 : : 5], y = history.history[f'val_{metric2.lower()}'][4 : : 5], color = 'orange', marker = 'o', s = 15, ax = ax[2])
    ax[2].axvline(x = len_histo - early, color = 'gray', linestyle = '--', label = 'Early Stopping')
    ax[2].set_xlabel('# Epochs')
    ax[2].set_ylabel(f'{metric2}')
    ax[2].legend(loc = 'lower right')
    #------- Show -------#
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
def evaluate_model_class(model, X_test, y_test, labels):
    '''
    Funcion para formar informar el desempeño del modelo de clasificacion
    a traves de un resumen con las distintas metricas de clasificacion y 
    junto a matrices de confusion para visualizar resultados del modelo de
    clasificacion de redes neuronales entrenado. Las matrices que se
    presentan es una matriz de confusion comun y otra ponderada por filas,
    para ver la tasa de acierto respecto a las clases reales.
    
    Inputs:
        model : modelo de red neuronal entrenado.
        X_test : imagenes a evaluar.
        y_test : etiquetas reales de las imagenes a evaluar.
        labels : lista con nombres reales de las etiquetas del problema.
        
    Outputs:
        Printea un resumen de las distintas metricas de clasificacion con el
        cual se evalua el modelo, y ademas grafica 2 matrices.
    '''
    #Prediccion
    y_pred = np.argmax(model.predict(X_test), axis = 1)
    #Reporte de metricas de clasificacion
    print(classification_report(y_test, y_pred))
    #Formacion de matriz de confusion y matriz ponderada
    conf_matrix = confusion_matrix(y_test, y_pred)
    pond_conf_matrix = pond_matriz(conf_matrix)
    #Figura de graficos
    fig, ax = plt.subplots(1, 2, figsize = (20, 6))
    #---------------------------- Graphs Pond Matrix ----------------------------#
    sns.heatmap(pond_conf_matrix, annot = True, fmt = '.2f', cmap = 'viridis', xticklabels = labels, yticklabels = labels, ax = ax[0])
    ax[0].set_ylabel('True Labels', fontsize = 16)
    ax[0].set_xlabel('Predict Labels', fontsize = 16)
    #---------------------------- Graphs Matrix ----------------------------#
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'viridis', xticklabels = labels, yticklabels = labels, ax = ax[1])
    ax[1].set_ylabel('True Labels', fontsize = 16)
    ax[1].set_xlabel('Predict Labels', fontsize = 16)
    #------- Show -------#
    fig.suptitle('Confusion Matrix', fontsize = 22)
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
def pond_matriz(matrix):
    '''
    Funcion que normaliza los valores de una matriz para que la suma por
    filas sea un total de 1 (representando el 100%).

    Inputs:
        matrix : np.array matriz cuadrada.

    Outputs:
        matrix : matriz ponderada.
    '''
    #Numero de filas de la matriz
    rows, _ = matrix.shape
    #Matriz con valores flotantes para poder dividir
    matrix = matrix.astype('float32')
    #Suma y normalizacion por filas
    for i in range(0, rows):
        suma = np.sum(matrix[i])
        matrix[i] = matrix[i] / suma
    return matrix
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
def list_dataframes_train(dataframe, columna, cant_dfs = 8):
    '''
    Funcion que divide un dataframe grande en una cantidad dada de
    dataframes mas pequeños, manteniendo la misma proporcionalidad de
    una columna dada

    Inputs:
        dataframe : dataframe grande el cual sera distribuido
        columna : columna del dataframe que se respetara proporcionalidad
        cant_dfs : numero de dataframes mas pequeños formados

    Outputs:
        list_data : lista con los dataframes pequeños que se formaron
    '''
    dataframe = dataframe[['imagen', columna]]
    class_0 = dataframe[dataframe[columna] == 0]
    class_zero = class_0.sample(frac = 0.875, random_state = 42)
    data_test_0 = class_0.drop(class_zero.index)
    class_1 = dataframe[dataframe[columna] == 1]
    class_one = class_1.sample(frac = 0.875, random_state = 42)
    data_test_1 = class_1.drop(class_one.index)
    class_2 = dataframe[dataframe[columna] == 2]
    class_two = class_2.sample(frac = 0.875, random_state = 42)
    data_test_2 = class_2.drop(class_two.index)
    data_test = pd.concat([data_test_0, data_test_1, data_test_2]).reset_index(drop = True)
    list_data = []
    for i in range(0, cant_dfs):
        data_aux = pd.concat([class_zero.iloc[: : cant_dfs, : :], class_one.iloc[: : cant_dfs, : :],
                              class_two.iloc[: : cant_dfs, : :]]).reset_index(drop = True)
        list_data.append(data_aux)
    list_data.append(data_test)
    return list_data
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
def prepare_data_image(data, columna, bucket, bucket_s3, flag_bgr = 0, percent_test = 0.15):
    X_entre, X_val, y_train, y_valid = train_test_split(data['imagen'], data[columna], test_size = percent_test, stratify = data[columna], random_state = 42)
    X_train, X_valid, indices_drop_tr, indices_drop_va = [], [], [], []
    for i, element in enumerate(X_entre):
        objeto = bucket_s3.get_object(Bucket = bucket, Key = element)
        image = Image.open(objeto['Body'])
        if image is None:
            indices_drop_tr.append(i)
        X_train.append(image)
    for j, elemento in enumerate(X_val):
        objet = bucket_s3.get_object(Bucket = bucket, Key = elemento)
        image = Image.open(objet['Body'])
        if image is None:
            indices_drop_va.append(j)
        X_valid.append(image)
    for i in indices_drop_tr:
        del X_train[i], y_train[i]
    for j in indices_drop_va:
        del X_valid[j], y_valid[j]
    X_train = [np.array(image.resize((224, 224), Image.ANTIALIAS)) for image in X_train]
    X_valid = [np.array(image.resize((224, 224), Image.ANTIALIAS)) for image in X_valid]
    if flag_bgr == 1:
        X_train = [image[:, :, [2, 1, 0]] for image in X_train]
        X_valid = [image[:, :, [2, 1, 0]] for image in X_valid]
    X_train = np.array([element.astype('float32') / 255 for element in X_train])
    X_valid = np.array([element.astype('float32') / 255 for element in X_valid])
    X_train = X_train.reshape(X_train.shape[0], 224, 224, 3)
    X_valid = X_valid.reshape(X_valid.shape[0], 224, 224, 3)
    y_train = to_categorical(y_train, num_classes = 3)
    y_valid = to_categorical(y_valid, num_classes = 3)
    return X_train, X_valid, y_train, y_valid
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#



