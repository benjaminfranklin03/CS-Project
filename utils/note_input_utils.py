# ===========================================================
# Imports
# ===========================================================
import uuid
import logging

import streamlit as st
from utils.graph_utils import save_graph, display_graph

# ===========================================================
# Logging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# Constant
# ===========================================================
GRAPH_FILE = "data/knowledge_graph.json"

# ===========================================================
# Note Input Forms
# ===========================================================
def render_note_input_forms(note_system):

    # Sidebar for note management
    st.sidebar.header("Note Management")
    if 'full_screen' not in st.session_state:
        st.session_state.full_screen = False

    def toggle_full_screen():
        """Toggle the full-screen mode."""
        st.session_state.full_screen = not st.session_state.full_screen

    st.sidebar.button("Toggle Full-Screen Mode", on_click=toggle_full_screen)

    # Note input forms
    if st.session_state.full_screen:
        # Full-Screen Note Input
        st.markdown("## 🖊️ Write Note (Full Screen)")
        with st.form("full_screen_note_form"):
            title = st.text_input("Note Title", key="full_screen_title")
            content = st.text_area("Note Content", height=500, key="full_screen_content")
            submit_button = st.form_submit_button("Add Note")

        if submit_button:
            if title and content:
                note_id = str(uuid.uuid4())
                try:
                    note_system.add_note(note_id, content, title)
                    cluster_id = note_system.notes[note_id].cluster_id or 0
                    st.session_state.knowledge_graph.add_node(title, cluster_id=cluster_id)
                    save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
                    st.success(f"Note added successfully! Title: {title}")
                except Exception as e:
                    st.error("An error occurred while adding the note.")
                    logger.error(f"Error while adding note: {e}")
            else:
                st.error("Please provide both a title and content.")
    else:
        # Standard Note Input Form in Sidebar
        st.sidebar.header("Add New Note")
        with st.sidebar.form("new_note_form"):
            title = st.text_input("Note Title", key="standard_title")
            content = st.text_area("Note Content", height=200, key="standard_content")
            submit_button = st.form_submit_button("Add Note")

            if submit_button:
                if title and content:
                    note_id = str(uuid.uuid4())
                    try:
                        note_system.add_note(note_id, content, title)
                        st.sidebar.success("Note added successfully!")
                        cluster_id = note_system.notes[note_id].cluster_id or 0
                        st.session_state.knowledge_graph.add_node(title, cluster_id=cluster_id)
                        logger.debug(f"Added node '{title}' with cluster_id {cluster_id}")
                        save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
                        if 'knowledge_graph_placeholder' in st.session_state:
                            display_graph(
                                st.session_state.knowledge_graph, 
                                st.session_state.knowledge_graph_placeholder
                            )
                    except ValueError as ve:
                        st.sidebar.error(str(ve))
                        logger.error(f"ValueError: {ve}")
                    except Exception as e:
                        st.sidebar.error("An error occurred while adding the note.")
                        logger.error(f"Exception: {e}")
                else:
                    st.sidebar.error("Please provide both a title and content.")