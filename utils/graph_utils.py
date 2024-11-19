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
# Loading the knowledge graph from a JSON file
# ===========================================================
@st.cache_data(show_spinner=False)
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
        return G
    else:
        logger.debug("Graph JSON file not found. Initializing empty graph.")
        return nx.Graph()

# ===========================================================
# Save the knowledge graph to a JSON file
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
    Display the knowledge graph using PyVis within a Streamlit app.

    Args:
        G: The NetworkX graph to display.
        graph_placeholder: The Streamlit placeholder where the graph will be rendered.
    """
    # Initialize PyVis Network with dark theme settings
    net = Network(height='750px', width='100%', bgcolor='#1e1e1e', font_color='white', directed=False, notebook=False)
    net.force_atlas_2based()  # Apply force-directed layout
    net.from_nx(G)  # Convert NetworkX graph to PyVis

    # Identify connected components and assign colors
    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    logger.info(f"Number of connected components: {num_components}")

    color_palette = px.colors.qualitative.Plotly
    if num_components > len(color_palette):
        color_palette *= (num_components // len(color_palette) + 1)

    component_color_map = {}
    for idx, component in enumerate(connected_components):
        color = color_palette[idx % len(color_palette)]
        for node in component:
            component_color_map[node] = color

    # Customize node appearance
    for node in net.nodes:
        node_id = node['id']
        color = component_color_map.get(node_id, '#00b4d8')  # Default color
        degree = G.degree(node_id)  # Get the degree of the node
        
        node.update({
            'shape': 'dot',
            'size': 18 + degree ,  # Base size is 18, scale by degree
            'color': {
                'background': color,
                'border': '#ffffff',
                'highlight': {'background': '#f9f871', 'border': '#ffffff'}
            },
            'title': f"Note: {node_id} \n Cluster: {node['cluster_id']}",  # Tooltip includes degree
            'font': {'color': '#ffffff', 'size': 16},
            'shadow': True
        })
        logger.debug(f"Node '{node_id}' assigned color {color} and size {18 + degree}")

    # Customize edge appearance
    for edge in net.edges:
        edge['color'] = {'color': '#555555', 'highlight': '#ffffff'}
        edge.update({
            'width': 1,  
            'smooth': {'type': 'straight'}  # Straight edges
        })

    # Enhanced interaction settings
    net.set_options('''
    {
        "nodes": {
            "shape": "dot",
            "size": 18,
            "font": {
                "size": 16,
                "color": "#ffffff"
            },
            "borderWidth": 0.5,
            "borderWidthSelected": 0.5,
            "shadow": {
                "enabled": true,
                "color": "#000000",
                "size": 10,
                "x": 0,
                "y": 0
            }
        },
        "edges": {
            "color": {
                "color": "#555555",
                "highlight": "#ffffff"
            },
            "width": 1,
            "smooth": {
                "type": "continuous",
                "forceDirection": "none",
                "roundness": 0.5
            }
        },
        "interaction": {
            "hover": true,
            "dragNodes": true,
            "zoomView": true,
            "dragView": true,
            "tooltipDelay": 200,
            "multiselect": true,
            "navigationButtons": true,
            "keyboard": true
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -30,   
                "centralGravity": 0.002,      
                "springLength": 400,           
                "springConstant": 0.001,       
                "damping": 0.95,                
                "avoidOverlap": 1              
            },
            "solver": "forceAtlas2Based",
            "timestep": 0.35,                  
            "stabilization": {
                "enabled": true,
                "iterations": 1000,            
                "updateInterval": 25
            }    
        },
        "layout": {
            "randomSeed": 42
        },
        "manipulation": {
            "enabled": false
        }
    }
    ''')

    # Generate and display the graph
    graph_html = net.generate_html()
    graph_html = graph_html.replace('<body>', '<body style="margin: 0; padding: 0; background-color: #1e1e1e;">')
    graph_html = graph_html.replace('width="100%"', 'width="100%" style="width: 100%; height: 100%;"')
    graph_html = graph_html.replace('height="750px"', 'height="100%" style="height:100%;"')

    with graph_placeholder:
        st.components.v1.html(graph_html, height=750, width=1000, scrolling=False)

# ===========================================================
# Removing a note from the knowledge graph
# ===========================================================
def remove_note_from_graph(note_title: str):
    """
    Remove a note from the knowledge graph by its title.

    Args:
        note_title: The title of the note to remove.
    """
    if note_title in st.session_state.knowledge_graph:
        st.session_state.knowledge_graph.remove_node(note_title)
        logger.info(f"Removed node '{note_title}' from the knowledge graph.")
    else:
        logger.warning(f"Node '{note_title}' not found in the knowledge graph.")