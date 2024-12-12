from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__) #crear aplicación

# Cargar el modelo
with open('transformers/mejor_modelo.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar los transformers
with open('transformers/transformer_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('transformers/transformer_target.pkl', 'rb') as f:
    target = pickle.load(f)

with open('transformers/transformer_one.pkl', 'rb') as f:
    one = pickle.load(f)

variables_one = ['Gender', 'ProductCategory']

@app.route("/") # decorador
def home(): # principal, funcion
    return jsonify({'mensaje': 'API de predicción en funcionamiento',
                   'endpoints': {'/predict': 'Usa este endpoint para realizar predicciones'}}) # siempre poner return

@app.route("/predict", methods = ['POST']) # decorador
def predict(): # principal, funcion
    try:
        data = request.get_json() # coge el json al usuario que nos da la info
        df_pred = pd.DataFrame(data, index=[0])
        print(df_pred)
        df_pred['DiscountsAvailed'] = df_pred['DiscountsAvailed'].astype('category')

        col_numericas = df_pred.select_dtypes(include=np.number).columns
        df_pred[col_numericas] = scaler.transform(df_pred[col_numericas])

        df_one = pd.DataFrame(one.transform(df_pred[variables_one]).toarray(), columns= one.get_feature_names_out()) #metodo para ponerle nombre
        df_pred = pd.concat([df_pred, df_one], axis=1)
        df_pred.drop(columns = variables_one, axis=1, inplace=True)

        df_pred = target.transform(df_pred)

        prediccion = model.predict(df_pred)
        probabilidad = model.predict_proba(df_pred)
        return jsonify({'prediccion': prediccion.tolist()[0],
                        'probabilidad': probabilidad.tolist()[0][1]})

    except:
        return jsonify({'respuestas':'Ha habido un problema en el envío de datos'})
    
if __name__ == '__main__':
    app.run(debug=True)

