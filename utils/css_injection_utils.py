# ===========================================================
# Imports
# ===========================================================
import streamlit as st

# ===========================================================
# Injecting CSS to customize the UI
# ===========================================================
def inject_global_css():
    """
    Find the abstracted customizable elements of streamlit and inject CSS to customize the UI to write directly into the html of the website
    """
    st.markdown(
        """
        <style>
        /* Boxes --> slightly darker background for contrast */
        .note-section {
            background-color: #242424; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 10px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); 
        }
        /* expanders: darker background */
        div[data-testid="stExpander"] {
            background-color: #242424; 
            border: none; /* No borders for a cleaner look */
            border-radius: 5px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); 
        }
        /* Expander Headers: text color and font */
        div[data-testid="stExpander"] .streamlit-expanderHeader {
            color: #EAEAEA; 
            font-family: 'Georgia', serif;
        }
        /* general button customization */
        .stButton>button, div[data-testid="stForm"] button {
            background-color: #343434; 
            color: #EAEAEA; 
            border: none; 
            border-radius: 5px; 
            padding: 8px 12px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); 
            transition: all 0.3s ease-in-out; /* hover effect */
        }
        /* Hover effect for buttons */
        .stButton>button:hover, div[data-testid="stForm"] button:hover {
            background-color: #454545; 
            transform: scale(1.05); /* scaling effect */
        }
        
        /* Titles --> Georgia, no shadows */
        h1, h2, h3 {
            font-family: 'Georgia', serif;
            color: #EAEAEA;
            text-shadow: none; 
        }  display: block; 
        

        /* Form container --> customizing the form's look */
        div[data-testid="stForm"] {
            background-color: #242424; 
            border-radius: 5px; 
            padding: 20px; 
            margin-bottom: 20px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); 
            border: 1px solid #343434; 
        }

        /* Form title styling */
        div[data-testid="stForm"] h3 {
            font-family: 'Georgia', serif;
            color: #EAEAEA; 
            margin-bottom: 10px;
            font-weight: bold;
        }

        /* Form input fields --> styling inputs within the form */
        div[data-testid="stForm"] input, div[data-testid="stForm"] textarea, div[data-testid="stForm"] select {
            background-color: #343434; 
            color: #EAEAEA; 
            border: 1px solid #454545; 
            border-radius: 5px; 
            padding: 10px; 
            font-family: 'Georgia', serif; 
        }

        /* Form input field --> highlight the input field when it's focused */
        div[data-testid="stForm"] input:focus, div[data-testid="stForm"] textarea:focus, div[data-testid="stForm"] select:focus {
            border: 1px solid #E1C16E; 
            outline: none; 
        }
        
        /* File uploader container */
        div[data-testid="stFileUploader"] {
        background-color: #242424; 
        border-radius: 5px; 
        padding: 15px; 
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); 
        border: 1px solid #343434; 
        }

        /* Uploader button: same css settings */
        div[data-testid="stFileUploader"] button {
            background-color: #343434; 
            color: #EAEAEA; 
            border: none; 
            border-radius: 5px; 
            padding: 8px 12px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); 
            transition: all 0.3s ease-in-out; 
        }

        /* Button Hover Effect for the Uploader Button */
        div[data-testid="stFileUploader"] button:hover {
            background-color: #454545; 
            transform: scale(1.05); 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )