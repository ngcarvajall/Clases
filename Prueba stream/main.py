import streamlit as st

st.set_page_config(
    page_title='Predicción de precios',
    page_icon='🏠',
    layout='centered'
)

st.title('Predicción de precios de casa con ML')

st.write('Si quieres predecir tu casa, estás en el lugar correcto')

st.markdown('Usa esto para **aplicaciones** de lo que vale tu casa') #formto mkd

# st.image(
#     #puedo crear la imagen y cargarla con la ruta relativa,
#     caption='Tu próxima casa',
#     use_column_width=True
# )
# # streamlit run main.py

# # preparar inputs oara el usuario

# col1,col2 = st.columns(2)
# with col1:
#     barrio = st.selectbox('Barrio', ['A', 'B', 'C', 'D'])
#     tipo_casa = st.selectbox('Barrio', ['A', 'B', 'C', 'D'])

# with col2:
#     habitaciones = st.number_input( 'habitaciones', min_value=1, max_value=7, step=1)
#     metros = st.number_input( 'habitaciones', min_value=1, max_value=7, step=1)

# diccionario_respuesta = {
#     'rooms': habitaciones,
#     'HouseType': tipo_casa,
#     'Neighborhood': barrio,
#     'Area': metros
# }
# st.write(barrio)

