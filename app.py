# ===========================================================
# Standard- and Third Party Library Imports
# ===========================================================
import logging
import streamlit as st

# ===========================================================
# Local Imports
# ===========================================================
from utils.css_injection_utils import inject_global_css
from utils.note_utils import NoteEmbeddingSystem
from utils.note_input_utils import render_note_input_forms
from utils.file_utils import handle_file_uploads
from utils.graph_utils import load_graph
from tabs.tab1 import render_tab1
from tabs.tab2 import render_tab2
from tabs.tab3 import render_tab3
from tabs.tab4 import render_tab4
from tabs.tab5 import render_tab5


# ===========================================================
# Configure Logging
# ===========================================================
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ===========================================================
# Constant
# ===========================================================
GRAPH_FILE = "cache/knowledge_graph.json"

# ===========================================================
# Initialize and Cache the NoteEmbeddingSystem
# ===========================================================
@st.cache_resource
def get_note_system():
    if 'note_system' not in st.session_state:
        st.session_state.note_system = NoteEmbeddingSystem()
    return st.session_state.note_system

# ===========================================================
# Main Function
# ===========================================================
def main():
    
    # page configuration
    logger.debug("Setting up page configuration")
    st.set_page_config(page_title="Notesidian", page_icon="💡", layout="wide")
    
    # injecting global CSS for customization
    logger.debug("Injecting global CSS")
    inject_global_css()
    
    st.title("💡 Notesidian") 

    # initializing note system and knowledge graph
    logger.debug("Initializing NoteEmbeddingSystem")
    note_system = get_note_system()
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = load_graph(GRAPH_FILE)

    # add current notes to the knowledge graph
    logger.debug("Adding notes to knowledge graph")
    for note_id, note in note_system.notes.items():
        cluster_id = note.cluster_id or 0
        st.session_state.knowledge_graph.add_node(note.title, cluster_id=cluster_id)

    # render note input forms
    render_note_input_forms(note_system)

    # handle file uploads
    handle_file_uploads(note_system)

    # =======================================================
    # Main Content Area with Tabs
    # =======================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cluster Visualization", "📝 My Notes", 
        "🔍 Similar Notes", "🌐 Knowledge Graph", "💬 Q&A"
    ])

    with tab1:
        render_tab1(note_system)

    with tab2:
        render_tab2(note_system)

    with tab3:
        render_tab3(note_system)

    with tab4:
        render_tab4(note_system)

    with tab5:
        render_tab5(note_system)

# ===========================================================
# Running the App
# ===========================================================
if __name__ == "__main__":
    main()


