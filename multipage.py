import streamlit as st

class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """
        self.pages.append({
          
                "title": title, 
                "function": func
            })

    def run(self):
        page = st.sidebar.selectbox(
            'Menú ', 
            self.pages, 
            format_func=lambda page: page['title']
        )
        page['function']()