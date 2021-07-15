# app de ML 
import streamlit as st
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
#import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad

from PIL import Image
import os 

#@st.cache(allow_output_mutation=True)
def app():
  def load_model():
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=(9,)))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    #adam = Adam(learning_rate=0.05, beta_1=b1_adam, beta_2=b2_adam, amsgrad=False, epsilon=1e-07, name="Adam")
    #return model, adam      
    return model

  def load_model_graph():
    inputs  = keras.Input(shape=(9,))
    dense1  = layers.Dense(9, activation='relu')(inputs)
    dense2  = layers.Dense(9, activation='relu')(dense1)
    outputs = layers.Dense(10, activation='sigmoid')(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs, name='glass_model')
    st.write(model)
    #dot_img_file = '/tmp/model_1.png'
    #keras.utils.plot_model(model, dot_img_file, show_shapes=False)
    return model
    
  def train_model_Adam():
    adam = Adam(learning_rate=learning_rate_adam, beta_1=b1_adam, beta_2=b2_adam, amsgrad=False, epsilon=1e-07, name="Adam")
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics = ['accuracy'])
    history_adam = model.fit(X_train, y_train, epochs=50,validation_split=0.1,verbose=2)
    #st.write('result:')
    test_loss_adam, test_acc_adam = model.evaluate(X_test, y_test)
    acc_models.append(test_acc_adam)
    loss_models.append(test_loss_adam)
    
    return history_adam, test_loss_adam, test_acc_adam

  def train_model_rmsprop():
    rmsprop = RMSprop(
      learning_rate=learning_rate_rmsp,
      rho=b1_rmsprp,
      momentum=0.0,
      epsilon=1e-07,
      centered=False,
      name="RMSprop")
    model.compile(loss="binary_crossentropy", optimizer=rmsprop, metrics = ['accuracy'])
    history_rmsprop = model.fit(X_train, y_train, epochs=50,validation_split=0.1,verbose=2)
    #st.write('result:')
    test_loss, test_acc= model.evaluate(X_test, y_test)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_rmsprop

  def train_model_Adadelta():
    adadelta_prop = Adadelta(
      learning_rate=learning_rate_adadelta,
      rho=Rho,
      epsilon=1e-07)
    model.compile(loss="binary_crossentropy", optimizer=adadelta_prop, metrics = ['accuracy'])
    history_adadelta = model.fit(X_train, y_train, epochs=50,validation_split=0.1,verbose=2)
    test_loss, test_acc= model.evaluate(X_test, y_test)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_adadelta

  def train_model_Adagrad():
    adagrad = Adagrad( learning_rate=learning_rate_adagrad, initial_accumulator_value=Vo, epsilon=1e-07,name="Adagrad",)
    model.compile(loss="binary_crossentropy", optimizer=adagrad, metrics = ['accuracy'])
    history_adagrad = model.fit(X_train, y_train, epochs=50,validation_split=0.1,verbose=2)
    test_loss, test_acc= model.evaluate(X_test, y_test)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_adagrad

  def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Dataset/data_glass.mat')
    data = sio.loadmat(data_dir)
    data = data['data']
    x = data[0::,0:-1]
    y = []
    for label in data[0::,-1]:  y.append(int(label)) 
    x = data[0:146,0:-1]
    y = []
    for label in data[0:146,-1]:  
      if int(label) == 2:
        y.append(0) 
      else:
        y.append(1)
    x = np.array(x)
    y = np.array(y)
    st.write('El tamaño de los datos biclase es: ',x.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

  ## Diseño web

  header = st.beta_container()
  st.markdown("""---""")  
  data_container = st.beta_container()
  st.markdown("""---""")  
  neural_model_container = st.beta_container()
  st.markdown("""---""")  

  with header:
    st.title('Comparación optimizadores redes neuronales')
    st.header('Ejemplo 1 - Clasificación de vidrios')
    #st.write('Esta interfaz tiene como proposito realizar un comparativo sencillo entre los optimizadores Adagrad, RMSprop, Adam y AdaDelta de la API de TensorFlow en la libreria Keras.')
    #st.write('En el contexto de las redes neuronales los optimizadores son algoritmos que permiten realizar el proceso de optimización de la función de perdidas, con el proposito de encontrar los pesos adecuados para las conexiones de las redes neuronales.')
  
  with data_container:
    st.header('Datos')
    st.write('El conjunto de datos para este experimento corresponde a una aplicación de clasificación de vidrios, se cuenta con 126 datos divididos en 6 clases de vidrios. El vector de características se compone de 9 atributos numéricos: RI: Índice de refracción, Na: Sodio presente en el vidrio, Mg: Magnesio, Al: Aluminio, Si: Silicon, K: Potasio, Ca: Calcio, Ba: Bario y Fe: Hierro. Base de datos disponible en: https://archive.ics.uci.edu/ml/datasets/glass+identification')
    
    #st.write('- RI: Indice de refracción')
    st.subheader('Etiquetas')
    st.write('Para simplificar el problema se trabajará un escenario de dos clases. Del conjunto de datos original se usarán los elementos pertenecientes a las clases 1 y 2.')
    st.write('- 1 vidrios para construcción de ventanas flotador procesado')
    st.write('- 2 vidrios para construcción de ventanas flotador no procesado')
    X_train, X_test, y_train, y_test = load_data()
    
    #st.write(X_test)
    flag_train = False
    #st.latex(r'''a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =\sum_{k=0}^{n-1} ar^k =a \left(\frac{1-r^{n}}{1-r}\right)''')

  with neural_model_container:
    st.header('Modelo Red Neuronal')
    image_dir = os.path.join(os.path.dirname(__file__), 'imagen_1.png')
    im = Image.open(image_dir)
    st.image(im)

  with st.form(key='columns_in_form'):
    st.header('Parámetros Optimizadores')
    c1, c2= st.beta_columns(2)
    with c1:
      st.subheader('Adagrad')
      learning_rate_adagrad = st.number_input("Tasa de aprendizaje adagrad", step=0.001,format="%.5f", min_value=0.00001)
      Vo = st.number_input("Acumulador inicial Vo", step=0.001,format="%.5f", min_value=0.00001)
    with c2:
      st.subheader('RMSprop')
      learning_rate_rmsp = st.number_input("Tasa de aprendizaje rms", step=0.001,format="%.5f", min_value=0.00001)
      b1_rmsprp = st.slider('b1_rms', 0.0, 1.0, 0.5)
    c3, c4= st.beta_columns(2)
    with c3:
      st.subheader('Adam')
      learning_rate_adam = st.number_input("Tasa de aprendizaje adam", step=0.001,format="%.5f", min_value=0.00001)
      b1_adam = st.slider('b1', 0.0, 1.0, 0.5)
      b2_adam = st.slider('b2', 0.0, 1.0, 0.5)
    with c4:
      st.subheader('Adadelta')
      learning_rate_adadelta = st.number_input("Tasa de aprendizaje adadelta", step=0.001,format="%.5f", min_value=0.00001)
      Rho = st.slider('Rho', 0.0, 1.0, 0.5)
      
    submitted = st.form_submit_button('Entrenar')
    
    if submitted:
      acc_models = []
      loss_models = []

      keras.backend.clear_session()
      model = load_model()
      history_adagrad = train_model_Adagrad()
      acc = history_adagrad.history['accuracy']
      val_acc = history_adagrad.history['val_accuracy'] 
      epochs  = range(1,len(acc)+1,1) 
      fig_adagrad, ax_adagrad = plt.subplots(figsize=(12, 5))
      ax_adagrad.plot  ( epochs, np.multiply(acc,100), '#F63366', label='Precisión Entrenamiento'  )
      ax_adagrad.plot  ( epochs, np.multiply(val_acc,100),  'g--', label='Precisión validación')
      ax_adagrad.set_title ('Precisión de entrenamiento y validación')
      ax_adagrad.set_ylabel('Precisión %')
      ax_adagrad.set_xlabel('Epoca')
      ax_adagrad.legend()

      keras.backend.clear_session()
      model = load_model()
      history_rmsprop = train_model_rmsprop()
      acc = history_rmsprop.history['accuracy']
      val_acc = history_rmsprop.history['val_accuracy']
      epochs  = range(1,len(acc)+1,1) 
      fig_rmsprop, ax_rmsprop = plt.subplots(figsize=(12, 5))
      ax_rmsprop.plot  ( epochs, np.multiply(acc,100), '#F63366', label='Precisión Entrenamiento'  )
      ax_rmsprop.plot  ( epochs, np.multiply(val_acc,100),  'g--', label='Precisión validación')
      ax_rmsprop.set_title ('Precisión de entrenamiento y validación')
      ax_rmsprop.set_ylabel('Precisión %')
      ax_rmsprop.set_xlabel('Epoca')
      ax_rmsprop.legend()

      keras.backend.clear_session()
      model = load_model()
      history_adam, test_loss_adam, test_acc_adam = train_model_Adam()
      acc = history_adam.history['accuracy']
      val_acc = history_adam.history['val_accuracy']
      epochs  = range(1,len(acc)+1,1) 
      fig_adam, ax_adam = plt.subplots(figsize=(12, 5))
      ax_adam.plot  ( epochs, np.multiply(acc,100), '#F63366', label='Precisión Entrenamiento'  )
      ax_adam.plot  ( epochs, np.multiply(val_acc,100),  'g--', label='Precisión validación')
      ax_adam.set_title ('Precisión de entrenamiento y validación')
      ax_adam.set_ylabel('Precisión %')
      ax_adam.set_xlabel('Epoca')
      ax_adam.legend()

      keras.backend.clear_session()
      model = load_model()
      history_adadelta = train_model_Adadelta()
      acc = history_adadelta.history['accuracy']
      val_acc = history_adadelta.history['val_accuracy']
      epochs  = range(1,len(acc)+1,1) 
      fig_adadelta, ax_adadelta = plt.subplots(figsize=(12, 5))
      ax_adadelta.plot  ( epochs, np.multiply(acc,100), '#F63366', label='Precisión Entrenamiento'  )
      ax_adadelta.plot  ( epochs, np.multiply(val_acc,100),  'g--', label='Precisión validación')
      ax_adadelta.set_title ('Precisión de entrenamiento y validación')
      ax_adadelta.set_ylabel('Precisión %')
      ax_adadelta.set_xlabel('Epoca')
      ax_adadelta.legend()

      flag_train = True

  st.markdown("""---""")  
  result_container = st.beta_container()
  with result_container:
    st.header('Resultados Entrenamiento')

  col1, col2= st.beta_columns(2)
  col3, col4 = st.beta_columns(2)

  with col1:
    if flag_train:
      st.subheader('Adagrad')
      st.pyplot(fig_adagrad)
  with col2:
    if flag_train:
      st.subheader('RMSprop')
      st.pyplot(fig_rmsprop)
  with col3:
    if flag_train:
      st.subheader('Adam')
      st.pyplot(fig_adam)
  with col4:
    if flag_train:
      st.subheader('Adadelta')
      st.pyplot(fig_adadelta)

  st.markdown("""---""")  
  eval_container = st.beta_container()

  with eval_container:
    st.header('Validación de modelos')
    if flag_train:
      #st.write(test_loss_adam)
      #st.write(eval_models)
      fig = go.Figure(data=[go.Table(header=dict(values=['Modelo','Acc','Loss'], 
                                      font=dict(size=16),
                                      fill_color='#F9809F'
                                    ), 
                                    cells=dict(values=[['AdaGrad', 'RMSprop', 'Adam', 'AdaDelta'], acc_models, loss_models],
                                    font=dict(size=16),
                                    height=35,
                                    fill_color='#F0F2F6'
                                    ),
                            )])
      fig.update_layout(margin=dict(l=5,r=25,t=10, b=10))
      st.write(fig)

