import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import main1, main2, optimizer

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.set_page_config(page_title='Optimizadores ML', layout="wide")
st.sidebar.image('logo.PNG', width=250)
#st.sidebar.title('Optimizadores redes neuronales')
st.sidebar.markdown('<div style="text-align: justify"> Esta interfaz tiene como proposito realizar un comparativo sencillo entre los optimizadores Adagrad, RMSprop, Adam y AdaDelta de la API de TensorFlow en la libreria Keras.  </div>', unsafe_allow_html=True)
st.sidebar.markdown("""---""")  
# Add all your applications (pages) here

app.add_page("Optimizadores", optimizer.app)
app.add_page("Ejemplo 1 - Clasificación de vidrios", main1.app)
app.add_page("Ejemplo 2 - Clasificación digitos MNIST", main2.app)
# The main app
app.run()