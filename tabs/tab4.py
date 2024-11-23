# ===========================================================
# Imports
# ===========================================================
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
GRAPH_FILE = "cache/knowledge_graph.json"

# ===========================================================
# Rendering the Knowledge Graph Tab
# ===========================================================
def render_tab4(note_system):
      
    st.header("Knowledge Graph")
    graph_placeholder = st.empty()
    display_graph(st.session_state.knowledge_graph, graph_placeholder)

    st.subheader("Add Connections Between Nodes")
    # Get the list of node titles
    node_titles = list(st.session_state.knowledge_graph.nodes())
    # Check if there are enough nodes to create connections
    if len(node_titles) < 2:
        st.info("Add more notes to create connections.")
    else:
        # Dropdowns to select source and target nodes within a form
        with st.form("add_connection_form"):
            source = st.selectbox("Select Source Node:", node_titles, key="source_node")
            target = st.selectbox("Select Target Node:", node_titles, key="target_node")
            connect_button = st.form_submit_button("Connect Nodes")

            if connect_button:
                if source == target:
                    st.error("Cannot connect a node to itself.")
                elif st.session_state.knowledge_graph.has_edge(source, target):
                    st.warning("These nodes are already connected.")
                else:
                    st.session_state.knowledge_graph.add_edge(source, target)
                    st.success(f"Connected **{source}** to **{target}**.")
                    logger.debug(f"Added edge between '{source}' and '{target}'")
                    # Save the updated graph
                    save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
                    # Refresh the graph
                    display_graph(st.session_state.knowledge_graph, graph_placeholder)

    st.markdown("---")
    st.subheader("Current Connections")
    if st.session_state.knowledge_graph.number_of_edges() > 0:
        for idx, (u, v) in enumerate(st.session_state.knowledge_graph.edges(), 1):
            st.write(f"{idx}. **{u}** ↔ **{v}**")
    else:
        st.info("No connections yet. Use the form above to add connections.")

    st.markdown("---")
    st.subheader("Remove Connections")
    if st.session_state.knowledge_graph.number_of_edges() == 0:
        st.info("No connections available to remove.")
    else:
        with st.form("remove_connection_form"):
            edge_options = [f"{u} ↔ {v}" for u, v in st.session_state.knowledge_graph.edges()]
            selected_edge = st.selectbox("Select Connection to Remove:", edge_options, key="remove_edge")
            remove_button = st.form_submit_button("Remove Connection")
            if remove_button and selected_edge:
                u, v = selected_edge.split(" ↔ ")
                st.session_state.knowledge_graph.remove_edge(u, v)
                st.success(f"Removed connection between **{u}** and **{v}**.")
                logger.debug(f"Removed edge between '{u}' and '{v}'")
                save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
                display_graph(st.session_state.knowledge_graph, graph_placeholder)