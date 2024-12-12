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
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    # for categoria in dataframe[columna_control].unique():
    #     dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
    #     print("\n ..................... \n")
    #     print(f"Los principales estadísticos de las columnas categóricas para el {categoria} son: ")
    #     display(dataframe_filtrado.describe(include = "O").T)
        
    #     print("\n ..................... \n")
    #     print(f"Los principales estadísticos de las columnas numéricas para el {categoria} son: ")
    #     display(dataframe_filtrado.describe().T)

def plot_outliers_univariados(dataframe, columnas_numericas, tipo_grafica, bins):
    fig, axes = plt.subplots(nrows=math.ceil(len(columnas_numericas) / 2), ncols=2, figsize= (15,10))

    axes = axes.flat

    for indice,columna in enumerate(columnas_numericas):

        if tipo_grafica.lower() == 'h':
            sns.histplot(x=columna, data=dataframe, ax= axes[indice], bins= bins)

        elif tipo_grafica.lower() == 'b':
            sns.boxplot(x=columna, 
                        data=dataframe, 
                        ax=axes[indice], 
                        # whis=whis, #para bigotes
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