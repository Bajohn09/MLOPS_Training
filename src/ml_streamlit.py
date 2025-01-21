
import time #Para la parte de sincronía y para medir el tiemp de procesamiento
import joblib

import pandas as pd
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import input.preprocessors as pp


from PIL import Image 
from conf.local import config


def segmentacion(df, pipeline_predict, pipeline_preprocess):
    
    pred_clusters = pipeline_predict.fit_predict(df)
    df_new = pipeline_preprocess.fit_transform(df)

    df_new = df_new[config.ID_COL + config.FEATURES]
    df_new.loc[:, "Cluster"] = pred_clusters

    return df_new, pred_clusters

def plot_variables(df, vars):
    num_subplots = len(vars)
    rows = (num_subplots + 1) // 2  # Calculate the number of rows for subplots
    
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows // 2+10))
    axes = axes.flatten()
    
    if num_subplots % 2 != 0:
        fig.delaxes(axes[-1]) 
        
    for i, var in enumerate(vars): 
        sns.barplot(data=df, x=df.index, y=var, hue=df.index, 
                    palette="Spectral", ax=axes[i])
        axes[i].set_title(var)
        
    plt.tight_layout()

    return fig


#Diseno de la Interface
st.title("Clusterizacion Facebook John Ortiz - DATAPATH")

image = Image.open('src/data/datapath-logo.png') #COMPLETAR CON UNA IMAGEN
st.image(image, use_container_width=True) #use_column_width esta "deprecated"

st.sidebar.write("Suba el archivo CSV de interacciones de Facebook para realizar la segmentación")


#------------------------------------------------------------------------------------------
# Cargar el archivo CSV desde la barra lateral
uploaded_file = st.sidebar.file_uploader(" ", type=['csv'])

if uploaded_file is not None:
    #Leer el archivo CSV y lo pasamos a Dataframe
    df_de_los_datos_subidos = pd.read_csv(uploaded_file)

    #Mostrar el contenido del archivo CSV
    st.write('Contenido del archivo CSV en formato Dataframe:')
    st.dataframe(df_de_los_datos_subidos)


#Cargar el Modelo ML o Cargar el Pipeline
pipeline_predict = joblib.load('src/data/03_models/facebook_predict_pipeline.joblib')
pipeline_preprocess = joblib.load('src/data/03_models/facebook_preprocess_pipeline.joblib')

if st.sidebar.button("click aqui para enviar el CSV al Pipeline"):
    if uploaded_file is None:
        st.sidebar.write("No se cargó correctamente el archivo, subalo de nuevo")
    else:
        with st.spinner('Pipeline y Modelo procesando...'):
            df_new, pred_clusters = segmentacion(df_de_los_datos_subidos, pipeline_predict, pipeline_preprocess)
            time.sleep(5)
            st.success('Listo!')

            # Mostramos los resultados de la predicción
            st.write('Resultados de la predicción:')
            st.write(df_new)

            #Graficar los clusters predichos
            sum_groups = df_new.groupby("Cluster").sum()

            status_fig = plot_variables(sum_groups, config.STATUS_VARS)
            
            st.pyplot(status_fig)

            num_fig = plot_variables(sum_groups, config.NUM_VARS)

            st.pyplot(num_fig)

            #Creamos el archivo CSV para descargar
            csv = df_new.to_csv(index=False).encode('utf-8')

            #Botón para descargar el CSV
            st.download_button(
                label="Descargar archivo CSV con segmentacion",
                data=csv,
                file_name='predicciones_clusters.csv',
                mime='text/csv',
            )
            

