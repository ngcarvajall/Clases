import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer, StandardScaler


def exploracion_dataframe(dataframe):
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
        print(f"La columna {col} tiene los siguientes valores únicos:")
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
    numericas = separar_dataframes(dataframe)[0]
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

def matriz_correlacion(dataframe, figsize = (5,3)):
    matriz_corr = dataframe.corr(numeric_only=True) #matriz correlación
    plt.figure(figsize=figsize)
    mascara = np.triu(np.ones_like(matriz_corr, dtype = np.bool_) )
    sns.heatmap(matriz_corr,
                annot=True,
                vmin=-1,
                vmax=1, 
                mask=mascara)
    plt.tight_layout()

def detectar_outliers(dataframe, color='red', tamaño_grafica=(15,10)):
    df_num = separar_dataframes(dataframe)[0]
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

def separar_dataframes(dataframe):
    return dataframe.select_dtypes(include = np.number), dataframe.select_dtypes(include = 'O')

def plot_numericas(dataframe, figsize=(10,8)):
    cols_numericas = dataframe.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows = num_filas, ncols = 2, figsize = figsize)
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
    plt.xticks(rotation=45)

def relacion_vr_numericas_problema_categorico(df, vr):
    df_num, df_cat = separar_dataframes(df)
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

def detectar_orden_cat(df, lista_cat, var_respuesta):
    lista_ordenadas = []
    lista_desordenadas = []
    for categorica in lista_cat:
        print(f'Estamos evaluando la variable {categorica.upper()}')
        df_cross_tab = pd.crosstab(df[categorica], df[var_respuesta])
        display(df_cross_tab)

        chi2, p, dof, excepted = chi2_contingency(df_cross_tab) 
        
        if p < 0.05:
            print(f'Sí tiene orden la variable {categorica}')
            lista_ordenadas.append(categorica)
        else:
            print(f'La variable {categorica} no tiene orden')
            lista_desordenadas.append(categorica)

    return lista_ordenadas, lista_desordenadas

def plot_relacion(self, vr, tamano_grafica=(40, 12)):


        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = False)
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "magma"
                              )
                
import seaborn as sns
import matplotlib.pyplot as plt
import math

def graficar_relaciones_categoricas(dataframe, vr):
    """
    Genera gráficos de barras en un subplot para todas las columnas categóricas
    mostrando su relación con la variable de respuesta.
    
    Args:
    - dataframe (pd.DataFrame): DataFrame que contiene los datos.
    - vr (str): Nombre de la variable de respuesta.
    """
    # Seleccionar columnas categóricas del DataFrame
    lista_cat = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Configurar el número de filas y columnas del subplot
    num_columnas = len(lista_cat)
    cols = 2  # Número de gráficos por fila
    filas = math.ceil(num_columnas / cols)  # Calcula las filas necesarias
    
    # Configurar la figura
    fig, axes = plt.subplots(nrows=filas, ncols=cols, figsize=(16, 6 * filas))
    axes = axes.flatten()  # Asegura que axes sea una lista para iterar fácilmente
    
    # Iterar sobre las columnas categóricas y crear gráficos
    for indice, columna in enumerate(lista_cat):
        sns.countplot(x=columna, hue=vr, data=dataframe, ax=axes[indice], palette="magma")
        axes[indice].set_title(f"Relación {columna} vs {vr}")
        axes[indice].set_xlabel(columna)
        axes[indice].set_ylabel("Frecuencia")
        axes[indice].legend(title=f"{vr}")
        axes[indice].tick_params(axis='x', rotation=45)
    
    # Ocultar cualquier gráfico sobrante si hay menos categorías que subplots
    for j in range(indice + 1, len(axes)):
        fig.delaxes(axes[j])  # Elimina subplots vacíos
    
    # Ajustar la disposición
    plt.tight_layout()
    plt.show()

def graficar_relaciones_numericas(dataframe, vr):
    """
    Genera gráficos de histogramas en un subplot para todas las columnas numéricas
    mostrando su relación con la variable de respuesta, omitiendo la variable de respuesta misma.
    
    Args:
    - dataframe (pd.DataFrame): DataFrame que contiene los datos.
    - vr (str): Nombre de la variable de respuesta.
    """
    # Seleccionar columnas numéricas del DataFrame
    lista_num = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Configurar el número de filas y columnas del subplot
    num_columnas = len(lista_num)
    cols = 2  # Número de gráficos por fila
    filas = math.ceil(num_columnas / cols)  # Calcula las filas necesarias
    
    # Configurar la figura
    fig, axes = plt.subplots(nrows=filas, ncols=cols, figsize=(16, 6 * filas))
    axes = axes.flatten()  # Asegura que axes sea una lista para iterar fácilmente
    
    # Contador para manejar los subplots correctamente
    plot_index = 0
    
    # Iterar sobre las columnas numéricas y crear gráficos
    for columna in lista_num:  # Itera solo sobre las columnas numéricas
        if columna == vr:  # Si la columna es la variable de respuesta, omitirla
            continue
        
        sns.histplot(x=columna, 
                     hue=vr, 
                     data=dataframe, 
                     ax=axes[plot_index], 
                     palette="magma", 
                     legend=True)  # Activa la leyenda
        axes[plot_index].set_title(f"Relación {columna} vs {vr}")
        axes[plot_index].set_xlabel(columna)
        axes[plot_index].set_ylabel("Frecuencia")
        axes[plot_index].tick_params(axis='x', rotation=45)
        plot_index += 1  # Incrementar el índice del subplot
    
    # Ocultar cualquier gráfico sobrante si hay menos variables que subplots
    for j in range(plot_index, len(axes)):
        fig.delaxes(axes[j])  # Elimina subplots vacíos
    
    # Ajustar la disposición
    plt.tight_layout()
    plt.show()

def visualizar_escalados_completos(df, lista_num):
    """
    Genera subplots para boxplots e histogramas de las columnas numéricas originales
    y sus versiones escaladas, cubriendo todas las columnas.
    
    Args:
    - df (pd.DataFrame): DataFrame con las columnas originales y escaladas.
    - lista_num (list): Lista de nombres de columnas numéricas originales.
    """
    # Sufijos para las versiones escaladas
    sufijos = ["robust", "minmax", "norm", "stand", ""]
    
    # Crear lista completa de columnas originales y escaladas
    columnas_totales = [f"{col}_{sufijo}" if sufijo else col for col in lista_num for sufijo in sufijos]
    
    # Número de columnas por fila en los subplots
    num_columnas = 5  # Máximo 5 gráficos por fila
    num_filas = math.ceil(len(columnas_totales) / num_columnas)  # Calcula filas necesarias
    
    # Crear figura
    fig, axes = plt.subplots(nrows=num_filas, ncols=num_columnas, figsize=(20, 5 * num_filas))
    axes = axes.flatten()  # Asegura que los ejes sean iterables
    
    # Iterar sobre las columnas para graficar
    for indice, col in enumerate(columnas_totales):
        if col in df.columns:  # Verifica que la columna exista en el DataFrame
            # Alternar entre boxplots e histogramas
            if "robust" in col or "minmax" in col or "norm" in col or "stand" in col:
                sns.boxplot(x=col, data=df, ax=axes[indice])
            else:
                sns.histplot(x=col, data=df, ax=axes[indice], bins=50)
            axes[indice].set_title(f"{col}")
        else:
            axes[indice].set_visible(False)  # Oculta subplots sin datos
    
    # Ocultar cualquier gráfico sobrante
    for j in range(len(columnas_totales), len(axes)):
        fig.delaxes(axes[j])
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()


def escalar_columnas(df, columnas_numericas):
    """
    Escala las columnas numéricas del DataFrame utilizando diferentes técnicas
    y agrega las columnas escaladas al DataFrame original.
    
    Args:
    - df (pd.DataFrame): DataFrame original.
    - columnas_numericas (list): Lista de nombres de columnas numéricas a escalar.
    
    Returns:
    - pd.DataFrame: DataFrame con las columnas escaladas añadidas.
    """
    # Inicializar escaladores
    escaladores = {
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "norm": Normalizer(),
        "stand": StandardScaler()
    }
    
    for nombre, escalador in escaladores.items():
        # Aplicar el escalador a las columnas numéricas
        datos_transformados = escalador.fit_transform(df[columnas_numericas])
        
        # Crear nombres para las columnas escaladas
        nombres_columnas = [f"{col}_{nombre}" for col in columnas_numericas]
        
        # Agregar las columnas transformadas al DataFrame original
        df[nombres_columnas] = datos_transformados
    
    return df

def escalar_columnas_metodo(df, columnas_numericas, metodo_escalador):
    """
    Escala directamente las columnas numéricas del DataFrame utilizando un método específico
    y guarda el escalador utilizado en un archivo pickle.
    
    Args:
    - df (pd.DataFrame): DataFrame original.
    - columnas_numericas (list): Lista de nombres de columnas numéricas a escalar.
    - metodo_escalador (str): Método de escalado a utilizar ('robust', 'minmax', 'norm', 'stand').
    - archivo_pickle (str): Nombre del archivo donde se guardará el escalador.
    
    Returns:
    - pd.DataFrame: DataFrame con las columnas numéricas escaladas.
    """
    # Inicializar los escaladores disponibles
    escaladores = {
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "norm": Normalizer(),
        "stand": StandardScaler()
    }
    
    # Verificar si el método de escalado es válido
    if metodo_escalador not in escaladores:
        raise ValueError(f"El método '{metodo_escalador}' no es válido. Usa uno de: {list(escaladores.keys())}")
    
    # Seleccionar el escalador
    scaler = escaladores[metodo_escalador]
    
    # Aplicar el escalado directamente a las columnas originales
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
    return df