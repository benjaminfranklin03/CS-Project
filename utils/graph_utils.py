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
def load_graph(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        G = nx.Graph()
        # Add nodes with attributes
        for node in data.get("nodes", []):
            G.add_node(node["id"], cluster_id=node.get("cluster_id", 0))
        # Add edges
        for edge in data.get("edges", []):
            G.add_edge(edge[0], edge[1])
        logger.debug("Graph loaded successfully from JSON.")
        # Set center nodes
        for node in data.get("center_nodes", []):
            if node in G:
                G.nodes[node]["center_node"] = True
            else:
                logger.warning(f"Center node '{node}' not found in the graph.")
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
        "edges": list(G.edges()),
        "center_nodes": [node for node, attrs in G.nodes(data=True) if attrs.get("center_node", False)]
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
    Displays the knowledge graph using PyVis with force-directed physics and straight edges.
    This configuration aims to create a layout similar to Obsidian's graph view.
    """

    # Initialize PyVis Network with dark theme settings
    net = Network(
        height="1000px", width="100%",
        bgcolor="#3C3C3C", font_color="white",
        directed=False, notebook=False
    )

    # Add nodes and edges from NetworkX graph to PyVis
    net.from_nx(G)
    
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

    # extracting center nodes from node attributes
    center_node_set = set(
        node for node, attrs in G.nodes(data=True) if attrs.get("center_node", False)
    )

    # customize node appearance
    for node in net.nodes:
        node_id = node["id"]
        cluster_color = component_color_map.get(node_id, "#C9A66E") 

        # determine styling for center nodes
        if node_id in center_node_set:
            background_color = cluster_color  # keeping cluster color for center nodes
            border_color = "#FFC300"  # gold border for center nodes
            size = 30  # larger size for center nodes
        else:
            background_color = cluster_color
            border_color = "#FFFFFF"
            size = 12 

        node.update({
            "shape": "dot",
            "size": size,
            "color": {
                "background": background_color,
                "border": border_color,
                "highlight": {"background": "#E1C16E", "border": "#E1C16E"},  #
            },
            "title": f"Node: {node_id}\nCluster: {G.nodes[node_id].get('cluster_id', 'N/A')}",
            "font": {"color": "#FFFFFF", "size": 18, "face": "Georgia"},
        })

    # Customize interaction settings
    net.set_options("""
    {
        "nodes": {
            "borderWidth": 1,
            "shadow": {
                "enabled": true,
                "color": "#000000",
                "size": 5
            }
        },
        "edges": {
            "color": {
                "color": "#AAAAAA",
                "highlight": "#FFD700"
            },
            "width": 0.5,
            "smooth": {
                "enabled": false
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
                "gravitationalConstant": -150,  
                "centralGravity": 0.005,
                "springLength": 120,  
                "springConstant": 0.08,
                "avoidOverlap": 1  
            },
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
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

    # Generate and display the graph
    graph_html = net.generate_html()
    graph_html = graph_html.replace(
        '<body>', 
        '<body style="margin: 0; padding: 0; background-color: #2D2D2D; display: flex; justify-content: center; align-items: center; height: 100vh;">'
    )

    with graph_placeholder:
        st.components.v1.html(graph_html, height=1024, width=1280, scrolling=False)


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
# Renaming a node in the knowledge graph when editing a note
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