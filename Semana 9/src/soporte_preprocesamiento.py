import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()))    
    
def relacion_vr_categoricas(dataframe, variable_respuesta, paleta = 'bright', tamaño_graficas = (15,10)):
    df_cat = separar_dataframe(dataframe)[1]
    cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows = num_filas, ncols = 2, figsize = tamaño_graficas)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta, ascending= False)
        display(datos_agrupados.head())
        sns.barplot(x=columna,
                    y = variable_respuesta,
                    data=datos_agrupados,
                    ax= axes[indice], 
                    palette=paleta)
        axes[indice].tick_params(rotation = 45)
        axes[indice].set_title(f'Relacion entre {columna} y {variable_respuesta}')
        axes[indice].set_xlabel('')

def relacion_numericas(dataframe, variable_respuesta, paleta = 'bright', tamaño_graficas = (15,10)):
    numericas = separar_dataframe(dataframe)[0]
    cols_numericas = numericas.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows = num_filas, ncols = 2, figsize = tamaño_graficas)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
            pass
        else:
            sns.scatterplot(x=columna,
                    y=variable_respuesta,
                    data=numericas,
                    ax=axes[indice], 
                    palette=paleta)
    plt.tight_layout()

def matriz_correacion(dataframe):
    matriz_corr = dataframe.corr(numeric_only=True) #matriz correlación
    plt.figure(figsize=(8,5))
    mascara = np.triu(np.ones_like(matriz_corr, dtype = np.bool_) )
    sns.heatmap(matriz_corr,
                annot=True,
                vmin=-1,
                vmax=1, 
                mask=mascara)
    plt.tight_layout()

def detectar_outliers(dataframe, color='red', tamaño_grafica=(15,10)):
    df_num = separar_dataframe(dataframe)[0]
    num_filas= math.ceil(len(df_num.columns)/2)
    fig, axes = plt.subplots(ncols=2, nrows=num_filas, figsize=tamaño_grafica)
    axes = axes.flat

    for indice,columna in enumerate(df_num.columns):
        sns.boxplot(x=columna,
                    data=df_num,
                    ax = axes[indice],
                    color=color)
        axes[indice].set_title(f'Outliers de {columna}')
        axes[indice].set_xlabel('')

def plot_outliers_univariados(dataframe, columnas_numericas, tipo_grafica, bins, whis=1.5):
    fig, axes = plt.subplots(nrows=math.ceil(len(columnas_numericas) / 2), ncols=2, figsize= (15,10))

    axes = axes.flat

    for indice,columna in enumerate(columnas_numericas):

        if tipo_grafica.lower() == 'h':
            sns.histplot(x=columna, data=dataframe, ax= axes[indice], bins= bins)

        elif tipo_grafica.lower() == 'b':
            sns.boxplot(x=columna, 
                        data=dataframe, 
                        ax=axes[indice], 
                        whis=whis, #para bigotes
                        flierprops = {'markersize': 2, 'markerfacecolor': 'red'})
        else:
            print('No has elegido grafica correcta')
    
        axes[indice].set_title(f'Distribucion columna {columna}')
        axes[indice].set_xlabel('')

    if len(columnas_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()

def identificar_outliers_iqr(dataframe,columnas_numericas ,k =1.5):
    diccionario_outliers = {}
    for columna in columnas_numericas:
        Q1, Q3 = np.nanpercentile(dataframe[columna], (25,75)) #esta no da problemas con nulos
        iqr = Q3 -Q1

        limite_superior = Q3 + (iqr * k)
        limite_inferior = Q1 - (iqr * k)

        condicion_superior = dataframe[columna] > limite_superior
        condicion_inferior = dataframe[columna] < limite_inferior

        df_outliers = dataframe[condicion_superior | condicion_inferior]
        print(f'La columna {columna.upper()} tiene {df_outliers.shape[0]} outliers')
        if not df_outliers.empty: #hacemos esta condicion por si acaso mi columna no tiene outliers
            diccionario_outliers[columna] = df_outliers

    return diccionario_outliers

def visualizar_categoricas(dataframe, lista_col_cat, variable_respuesta, bigote=1.5, paleta = 'bright',tipo_grafica='boxplot', tamaño_grafica=(15,10), metrica_barplot = 'mean',):
    num_filas = math.ceil(len(lista_col_cat)/ 2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamaño_grafica)

    axes = axes.flat

    for indice, columna in enumerate(lista_col_cat):
        if tipo_grafica.lower()=='boxplot':
            sns.boxplot(x=columna, 
                        y=variable_respuesta, 
                        data=dataframe,
                        whis=bigote,
                        hue=columna,
                        legend=False,
                        ax= axes[indice])
            
        elif tipo_grafica.lower()== 'barplot':
            sns.barplot(x=columna,
                        y=variable_respuesta,
                        ax = axes[indice],
                        data=dataframe,
                        estimator=metrica_barplot,
                        palette= paleta)
        else:
            print('No has elegido una grafica correcta')

        axes[indice].set_title(f'Relacion {columna} con {variable_respuesta}')
        axes[indice].set_xlabel('')
        plt.tight_layout()

def separar_dataframe(dataframe):
    return dataframe.select_dtypes(include = np.number), dataframe.select_dtypes(include = 'O')

def plot_numericas(dataframe):
    cols_numericas = dataframe.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows = num_filas, ncols = 2, figsize = (15,5))
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        sns.histplot(x=columna, data=dataframe, ax = axes[indice], bins=50)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel('')
    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()





def plot_categoricas(dataframe, paleta="bright", tamano_grafica=(15, 8)):
    """
    Grafica la distribución de las variables categóricas del DataFrame.

    Parameters:
    - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
    - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
    """
    # dataframe_cat = self.separar_dataframe()[1]
    _, axes = plt.subplots(2, math.ceil(len(dataframe.columns) / 2), figsize=tamano_grafica)
    axes = axes.flat
    for indice, columna in enumerate(dataframe.columns):
        sns.countplot(x=columna, data=dataframe, order=dataframe[columna].value_counts().index,
                        ax=axes[indice], palette=paleta)
        axes[indice].tick_params(rotation=0)
        axes[indice].set_title(columna)
        axes[indice].set(xlabel=None)

    plt.tight_layout()
    plt.suptitle("Distribución de variables categóricas")

def relacion_vr_numericas_problema_categorico(df, vr):
    df_num, df_cat = separar_dataframe(df)
    columnas_numericas = df_num.columns
    num_filas = math.ceil(len(columnas_numericas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=(15,10))
    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):
        sns.histplot(df, x=columna, ax=axes[indice], hue=vr , bins=20)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(columnas_numericas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()