import logging
import uuid

import streamlit as st
import PyPDF2

from utils.graph_utils import save_graph, display_graph

logger = logging.getLogger(__name__)

# ===========================================================
# Constants
# ===========================================================
GRAPH_FILE = "data/knowledge_graph.json"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = ["txt", "pdf"]

# ===========================================================
# File Uploading
# ===========================================================
def handle_file_uploads(note_system):
    """Handle file uploads and add them as notes."""
    st.sidebar.header("Upload Notes as Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF and Text files (.pdf, .txt)", 
        type=ALLOWED_EXTENSIONS, 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # File validation and content extraction
            if uploaded_file.size > MAX_FILE_SIZE:
                st.sidebar.error(f"File {uploaded_file.name} is too large. Maximum allowed size is 5 MB.")
                continue
            file_extension = uploaded_file.name.split('.')[-1].lower()
            content = ""

            if file_extension == "pdf":
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"
                    content = content.strip()
                    if not content:
                        st.sidebar.warning(f"No text found in {uploaded_file.name}. Skipping.")
                        continue
                except Exception as e:
                    st.sidebar.error(f"Failed to read PDF {uploaded_file.name}: {e}")
                    logger.error(f"Error reading PDF {uploaded_file.name}: {e}")
                    continue
            elif file_extension == "txt":
                try:
                    content = uploaded_file.read().decode("utf-8").strip()
                except Exception as e:
                    st.sidebar.error(f"Failed to read {uploaded_file.name}: {e}")
                    logger.error(f"Error reading TXT {uploaded_file.name}: {e}")
                    continue
            else:
                st.sidebar.error(f"Unsupported file type for {uploaded_file.name}.")
                continue

            # Add the extracted content as a note
            title = uploaded_file.name.rsplit(".", 1)[0]
            if not content:
                st.sidebar.warning(f"File {uploaded_file.name} is empty. Skipping.")
                continue
            note_id = str(uuid.uuid4())

            try:
                note_system.add_note(note_id, content, title)
                st.sidebar.success(f"Uploaded and added {uploaded_file.name} as a note.")
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
                st.sidebar.error(f"Error adding {uploaded_file.name}: {ve}")
                logger.error(f"ValueError adding {uploaded_file.name}: {ve}")
            except Exception as e:
                st.sidebar.error(f"An error occurred while adding {uploaded_file.name}.")
                logger.error(f"Exception adding {uploaded_file.name}: {e}")