# ===========================================================
# Imports
# ===========================================================
import os
import json
import logging

import networkx as nx
import streamlit as st
from pyvis.network import Network
import plotly.express as px

# ===========================================================
# Logging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# Loading the knowledge graph from the JSON file
# ===========================================================
@st.cache_data(show_spinner=False)
def load_graph(file_path):
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        G = nx.Graph()
        # add nodes with attributes
        for node in data.get("nodes", []):
            G.add_node(node["id"], cluster_id=node.get("cluster_id", 0))
        # add edges
        for edge in data.get("edges", []):
            G.add_edge(edge[0], edge[1])
        logger.debug("Graph loaded successfully from JSON.")
        return G
    else:
        logger.debug("Graph JSON file not found. Initializing empty graph.")
        return nx.Graph()

# ===========================================================
# Saving the knowledge graph to the JSON file
# ===========================================================
def save_graph(G, file_path):
    
    data = {
        "nodes": [{"id": node, "cluster_id": attrs.get("cluster_id", 0)} for node, attrs in G.nodes(data=True)],
        "edges": list(G.edges())
    }
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.debug("Graph saved successfully to JSON.")
    except Exception as e:
        st.error(f"Failed to save the knowledge graph: {e}")
        logger.error(f"Error saving graph: {e}")

# ===========================================================
# Displaying knowledge graph
# ===========================================================
def display_graph(G, graph_placeholder):
    """
    Displays the customized knowledge graph using PyVis
    """
    # Initialize PyVis Network with improved dark theme settings
    net = Network(height="740px", width="100%", bgcolor="#3C3C3C", font_color="white", directed=False, notebook=False)
    net.force_atlas_2based()  # apply force-directed layout for smoother spacing
    net.from_nx(G)  # convert NetworkX graph to PyVis

    # identify connected components and assign unique colors
    connected_components = list(nx.connected_components(G))
    color_palette = px.colors.qualitative.Vivid  # brighter colors
    if len(connected_components) > len(color_palette):
        color_palette *= (len(connected_components) // len(color_palette) + 1)  

    component_color_map = {
        node: color_palette[idx % len(color_palette)]
        for idx, component in enumerate(connected_components)
        for node in component
    }

    # customize node appearance
    for node in net.nodes:
        node_id = node["id"]
        degree = G.degree[node_id]  # node degree (i.e. number of edges)
        cluster_color = component_color_map.get(node_id, "#C9A66E")  # default to gold if no color assigned

        node.update({
            "shape": "dot",
            "size": 15 + degree,  # scale size by the number of connections
            "color": {
                "background": cluster_color,
                "border": "#FFFFFF",
                "highlight": {"background": "#E1C16E", "border": "#E1C16E"},  # gold for hovering
            },
            "title": f"Node: {node_id}\nCluster: {node.get('cluster_id', 'N/A')}",
            "font": {"color": "#FFFFFF", "size": 14}
        }) 


    # customized interaction settings
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 16,
                "color": "#FFFFFF"
            },
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "shadow": {
                "enabled": true,
                "color": "#000000",
                "size": 10
            }
        },
        "edges": {
            "color": {
                "color": "#555555",
                "highlight": "#E1C16E"
            },
            "width": 0.5,
            "smooth": {
                "type": "dynamic",
                "forceDirection": "none",
                "roundness": 0.5
            }
        },
        "interaction": {
            "hover": true,
            "dragNodes": true,
            "zoomView": true,
            "dragView": true,
            "tooltipDelay": 200
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.006,
                "springLength": 200,
                "springConstant": 0.005,
                "damping": 0.8,
                "avoidOverlap": 0.75
            },
            "solver": "forceAtlas2Based",
            "timestep": 0.4,
            "stabilization": {
                "enabled": true,
                "iterations": 500,
                "updateInterval": 25
            }
        },
        "layout": {
            "randomSeed": 42
        }
    }
    """)
    
    # generate and display the graph
    graph_html = net.generate_html()
    graph_html = graph_html.replace(
    '<body>', 
    '<body style="margin: 0; padding: 0; background-color: #2D2D2D; display: flex; justify-content: center; align-items: center; height: 100vh;">'
    )

    with graph_placeholder:
        st.components.v1.html(graph_html, height=750, width=1000, scrolling=False)

# ===========================================================
# Removing a note from the knowledge graph
# ===========================================================
def remove_note_from_graph(note_title: str):
    """
    Remove a note from the knowledge graph by its title

    note_title: The title of the note to remove
    """
    if note_title in st.session_state.knowledge_graph:
        st.session_state.knowledge_graph.remove_node(note_title)
        logger.info(f"Removed node '{note_title}' from the knowledge graph.")
    else:
        logger.warning(f"Node '{note_title}' not found in the knowledge graph.")

# ===========================================================
# Renaming a node int the knowledge graph when editing a note
# ===========================================================
def rename_node_in_graph(G, old_title: str, new_title: str):
    """
    Rename a node in the graph from old_title to new_title

    Args:
        G: knowledge graph
        old_title: the current title of the node
        new_title: the new title of the node
    """
    if old_title in G:
        mapping = {old_title: new_title}
        nx.relabel_nodes(G, mapping, copy=False)
        logger.info(f"Renamed node '{old_title}' to '{new_title}' in the knowledge graph.")
    else:
        logger.warning(f"Node '{old_title}' not found in the knowledge graph.")
