import streamlit as st
def app():
  optimizers_container = st.beta_container()

  with optimizers_container:
      st.title('Comparación optimizadores redes neuronales')
      st.header('Modelo matemático')
      st.write('En el contexto de las redes neuronales los optimizadores son algoritmos que permiten realizar el proceso de optimización de la función de pérdidas, con el propósito de encontrar los pesos adecuados para las conexiones de las redes neuronales. ')
      st.markdown("""---""")  
      col_adagrad, col_rmsprop= st.beta_columns(2)
      with col_adagrad:
        st.subheader('Adagrad')
        st.markdown('<div style="text-align: justify"> Intuitivamente el método disminuye la tasa de aprendizaje en función de los valores históricos del cuadrado del gradiente. Cuanto menor sea el gradiente acumulado, menor será el valor de v , lo que conducirá a una mayor tasa de aprendizaje. Después de varias iteraciones, los parámetros frecuentes comenzarán a recibir actualizaciones muy pequeñas debido a la disminución de la tasa de aprendizaje. </div>', unsafe_allow_html=True)
        st.latex(r'''v_t =  v_{t-1}+  \nabla w_t^2''')
        st.latex(r'''w_{t+1} = w_t-\frac{\eta}{\sqrt{v_t + \epsilon}} \nabla w_t''')
      with col_rmsprop:
        st.subheader('RMSprop')
        st.markdown('<div style="text-align: justify"> El método toma la filosofía del algoritmo AdaGram con la diferencia que incluye un término que evita la acumulación excesiva de los valores del cuadrado del gradiente. La acumulación excesiva se corrige con un promedio exponencial ponderado, este promedio se pondera según la proximidad de los valores históricos.</div>', unsafe_allow_html=True)
        st.latex(r'''v_t = \beta v_{t-1}+ (1-\beta) \nabla w_t^2''')
        st.latex(r'''w_{t+1} = w_t-\frac{\eta}{\sqrt{v_t + \epsilon}} \nabla w_t''')
      col_adam, col_adadelta= st.beta_columns(2)
      with col_adam:
        st.subheader('Adam')
        st.markdown('<div style="text-align: justify"> Desde el enfoque físico el algoritmo puede interpretarse como incluir los factores de momento y fricción. La idea física es "soltar una bola pesada por la superficie (función a optimizar) y tener en cuenta que se desliza por la mayor pendiente (gradiente) pero se tiene en cuenta la inercia y la fricción El algoritmo emplea como primer momento el promedio de los gradientes  (inercia) y segundo momento la variancia (fricción). Los pesos se actualizan en función del decaimiento exponencial de ambos momentos. </div>', unsafe_allow_html=True)
        st.latex(r'''m_t = \beta_1 m_{t-1}+ (1-\beta_1) \nabla w_t''')
        st.latex(r'''v_t = \beta_2 v_{t-1}+ (1-\beta_2) \nabla w_t^2''')
        st.latex(r'''m_t^* = \frac{m_t}{1-\beta_1^t} \ v_t^* = \frac{v_t}{1-\beta_2^t}''')
        st.latex(r'''w_{t+1} = w_t-\frac{\eta}{\sqrt{v_t^* + \epsilon}} m_t^* ''')
      with col_adadelta:
        st.subheader('Adadelta')
        st.markdown('<div style="text-align: justify"> En Adadelta se realiza una actualización teniendo en cuenta el valor RMS de un promedio exponencial ponderado de los valores cuadrados del gradiente. Este algoritmo de optimización parte de la idea básica detrás de incluir una función de corrección de desajuste de unidades entre los valores de actualización y los parámetros originales. No es necesario fijar una tasa de aprendizaje, sin embargo el algoritmo de Keras si lo permite, el método se beneficia de valores altos. La constante h se incializa por defecto en 0. </div>', unsafe_allow_html=True)
        st.latex(r'''v_t = \rho v_{t-1}+ (1-\rho) \nabla w_t^2''')
        st.latex(r'''s_t = \rho s_{t-1}+ (1-\rho) \nabla h_t^2''')
        st.latex(r'''h_t = \frac{\sqrt{s_t+\epsilon}}{\sqrt{v_t+\epsilon}} \nabla w_t''')
        st.latex(r'''w_{t+1} = w_t-h_t''')
