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
    #create a placeholder and display the graph in it
    graph_placeholder = st.empty()
    display_graph(st.session_state.knowledge_graph, graph_placeholder)

    st.subheader("Add Connections Between Nodes")
    
    # get the list of node titles
    node_titles = list(st.session_state.knowledge_graph.nodes())
    
    # check if there are enough nodes to create connections
    if len(node_titles) < 2:
        st.info("Add more notes to create connections.")
        
    else:
        # dropdowns to select source and target nodes within a form
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
                    
                    # save the updated graph
                    save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
                    
                    # refresh the graph
                    display_graph(st.session_state.knowledge_graph, graph_placeholder)

    # set center nodes
    st.subheader("Set Center Nodes")
    
    with st.form("set_center_nodes_form"):
        
        # get the list of node titles
        node_titles = list(st.session_state.knowledge_graph.nodes())
        
        # extract current center nodes from node attributes
        current_center_nodes = [
            node for node, attrs in st.session_state.knowledge_graph.nodes(data=True)
            if attrs.get("center_node", False)
        ]
        
        # multi-select to choose center nodes
        center_nodes = st.multiselect(
            "Select Center Nodes:",
            node_titles,
            default=current_center_nodes,
            key="center_nodes_selection"
        )
        
        set_center_button = st.form_submit_button("Set Center Nodes")

        if set_center_button:
            
            # first, reset all nodes' center_node attribute
            for node in st.session_state.knowledge_graph.nodes():
                
                if node in center_nodes:
                    st.session_state.knowledge_graph.nodes[node]["center_node"] = True
                else: 
                    st.session_state.knowledge_graph.nodes[node]["center_node"] = False
                    
            st.success(f"Center nodes set to: {', '.join(center_nodes) if center_nodes else 'None'}")
            logger.debug(f"Set center nodes to: {center_nodes}")
            
            # save the updated graph
            save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
            
            # force rerun, since the graph and session state have changed
            st.rerun()

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