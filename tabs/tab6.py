# ===========================================================
# Imports
# ===========================================================
import logging
import streamlit as st
from huggingface_hub import InferenceApi

# ===========================================================
# Logging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# Initializing Hugging Face Inference API
# ===========================================================

# ===========================================================
# Rendering the Q&A Tab
# ===========================================================
def render_tab6(notesystem):
    st.header("Q&A")
    #semantic search to get top3 notes
    #get summaries and topics and add them to the query (i.e. question:... \n Based on the following retrieved summaries, provide a detailed response: \n\n topic: ... \n summary: ...\n\n topic: ... \n summary: ...\n\n topic: ... \n summary: ...)
    