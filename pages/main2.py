import streamlit as st
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers


from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad



def app():
  
  def load_model():
    keras.backend.clear_session()
    inputs  = keras.Input(shape=(784,))
    dense1  = layers.Dense(50, activation='relu')(inputs)
    dense2  = layers.Dense(50, activation='relu')(dense1)
    dense3  = layers.Dense(20, activation='relu')(dense2)
    dense4  = layers.Dense(20, activation='relu')(dense3)
    outputs = layers.Dense(10, activation='sigmoid')(dense4)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    return model

  def train_model_Adagrad():
    adagrad = Adagrad( learning_rate=learning_rate_adagrad, initial_accumulator_value=Vo, epsilon=1e-07,name="Adagrad",)
    model.compile(loss="categorical_crossentropy", optimizer=adagrad, metrics = ['accuracy'])
    history_adagrad = model.fit(X_train, y_train_onehot, batch_size=10, epochs=20, validation_split=0.3,verbose=2)
    test_loss, test_acc= model.evaluate(X_test, y_test_onehot)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_adagrad

  def train_model_rmsprop():
    rmsprop = RMSprop(
    learning_rate=learning_rate_rmsp,
    rho=b1_rmsprp,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop")
    model.compile(loss="categorical_crossentropy", optimizer=rmsprop, metrics = ['accuracy'])
    history_rmsprop = model.fit(X_train, y_train_onehot, batch_size=10, epochs=20, validation_split=0.3,verbose=2)
    test_loss, test_acc= model.evaluate(X_test, y_test_onehot)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_rmsprop

  def train_model_Adam():
    adam = Adam( learning_rate=learning_rate_adam, beta_1=b1_adam, beta_2=b2_adam, amsgrad=False, epsilon=1e-07, name="Adam")
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics = ['accuracy'])
    history_adam = model.fit(X_train, y_train_onehot, batch_size=5, epochs=20, validation_split=0.3,verbose=2)
    test_loss, test_acc= model.evaluate(X_test, y_test_onehot)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_adam

  def train_model_Adadelta():
    adadelta_prop = Adadelta(
        learning_rate=learning_rate_adadelta,
        rho=Rho,
        epsilon=1e-07)
    model.compile(loss="categorical_crossentropy", optimizer=adadelta_prop, metrics = ['accuracy'])
    history_adadeltaprop = model.fit(X_train, y_train_onehot, batch_size=10, epochs=20, validation_split=0.3,verbose=2)
    test_loss, test_acc= model.evaluate(X_test, y_test_onehot)
    acc_models.append(test_acc)
    loss_models.append(test_loss)
    return history_adadeltaprop
  
  def load_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train=X_train.reshape(len(y_train),784)
    X_test=X_test.reshape(len(y_test),784)
    X_train=((X_train/255.0)-0.5)**2
    X_test=((X_test/255.0)-0.5)**2
    X_train_centered=X_train
    X_test_centered=X_test

    y_train_onehot = keras.utils.to_categorical(y_train)
    y_test_onehot =  keras.utils.to_categorical(y_test)

    del X_train, X_test
    return X_train_centered, X_test_centered, y_train_onehot, y_test_onehot, y_train
  
  header = st.beta_container()
  st.markdown("""---""")
  data_container = st.beta_container()
  st.markdown("""---""")
  neural_model_container = st.beta_container()
  st.markdown("""---""")
  
  with header:
    st.title('Comparación optimizadores redes neuronales')
    st.header('Ejemplo 2 - Clasificación digitos MNIST')
  
  with data_container:
    st.header('Datos')
    st.markdown('<div style="text-align: justify"> El conjunto de datos de dígitos del MNIST (Instituto Nacional de Estándares y Tecnología) es un repositorio que contiene  dígitos manuscritos. Cada imagen tiene 28 x 28 píxeles para un total de 784 píxeles. Los datos ya están previamente divididos en entrenamiento y validación destinando 60000 y 10000 datos, respectivamente. </div>', unsafe_allow_html=True)
    #st.write('- RI: Indice de refracción')
    
    X_train, X_test, y_train_onehot, y_test_onehot, y_train = load_data()

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,figsize=(10, 5),)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='bone')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader('Etiquetas')
    st.write('Las etiquetas están codificadas con números de 0 a 9. Para trabajar de manera adecuada es necesario llevar los valores enteros a una codificación one-hot. ')
    y_train_string = '['
    for x in y_train[:5]:
      y_train_string += str(x) + ', '
    st.write('Primeras cinco etiquetas: ', y_train_string + ']')
    st.write('Primeras cinco etiquetas codificación (one-hot):')

    for x in y_train_onehot[:5]:
      y_train_onehot_string = '['
      for y in x:
        y_train_onehot_string += str(int(y))
      st.write('- '+ y_train_onehot_string + ']')
    
    flag_train = False
  
  with neural_model_container:
    st.header('Modelo Red Neuronal')
    st.image('imagen_2.png')


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
      print('OK')
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
      history_adam = train_model_Adam()
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