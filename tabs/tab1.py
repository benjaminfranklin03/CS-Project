# ===========================================================
# Imports
# ===========================================================

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# ===========================================================
# Rendering the Cluster Visualization Tab
# ===========================================================

def render_tab1(note_system):
    st.header("Note Clusters Visualization")
    
    if len(note_system.notes) >= 2:
        clusters = note_system.cluster_notes()
        coords = note_system.get_2d_visualization()

        # Prepare DataFrame
        data = []
        for note_id, (x, y) in coords.items():
            note = note_system.notes[note_id]
            data.append({
                'x': x,
                'y': y,
                'title': note.title,
                'cluster': f'Cluster {note.cluster_id}',
                'content': note.content[:100] + '...' if len(note.content) > 100 else note.content,
                'is_centroid': note.is_centroid
            })

        df = pd.DataFrame(data)

        # Separate dataframes for centroids and non-centroids
        df_centroids = df[df['is_centroid']]
        df_non_centroids = df[~df['is_centroid']]

        # Create figure
        fig = go.Figure()

        # Define color mapping for clusters
        clusters_list = df['cluster'].unique()
        cluster_colors = px.colors.qualitative.Plotly
        color_map = {cluster: cluster_colors[i % len(cluster_colors)] for i, cluster in enumerate(clusters_list)}

        # Add non-centroid notes
        fig.add_trace(go.Scatter(
            x=df_non_centroids['x'],
            y=df_non_centroids['y'],
            mode='markers',
            name='',  # No legend entry
            showlegend=False,
            marker=dict(
                symbol='circle',
                size=10,
                color=df_non_centroids['cluster'].map(color_map),
            ),
            text=df_non_centroids.apply(lambda row: f"Title: {row['title']}<br>Cluster: {row['cluster']}", axis=1),
            hoverinfo='text',
        ))

        # Add centroids
        fig.add_trace(go.Scatter(
            x=df_centroids['x'],
            y=df_centroids['y'],
            mode='markers',
            name='Centroid',
            marker=dict(
                symbol='diamond',
                size=12,
                line=dict(width=2, color='DarkSlateGrey'),
                color=df_centroids['cluster'].map(color_map),
            ),
            text=df_centroids.apply(lambda row: f"Title: {row['title']}<br>Cluster: {row['cluster']}", axis=1),
            hoverinfo='text',
        ))

        # Update layout
        fig.update_layout(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            legend_title='Cluster and Centroid',
            hovermode='closest',
            showlegend=True
        )

        # manually adding cluster legend entries
        for cluster, color in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=cluster,
                showlegend=True,
                name=cluster
            ))

        # adjusting legend order
        fig.update_layout(legend=dict(
            itemsizing='constant'
        ))

        # added centroid annotations
        annotations = [
            dict(
                x=row['x'],
                y=row['y'],
                text=row['title'],
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20
            ) for _, row in df_centroids.iterrows()
        ]
        fig.update_layout(annotations=annotations)

        st.plotly_chart(fig, use_container_width=True)

        # displaying cluster information
        st.subheader("Cluster Information")
        for cluster_id, note_ids in clusters.items():
            with st.expander(f"Cluster {cluster_id} ({len(note_ids)} notes)"):
                cluster_notes = [note_system.notes[n_id] for n_id in note_ids]
                for note in cluster_notes:
                    centroid_marker = " (Centroid)" if note.is_centroid else ""
                    st.markdown(f"**{note.title}{centroid_marker}**")
                    truncated_content = (note.content[:200] + '...') if len(note.content) > 200 else note.content
                    st.write(truncated_content)
    elif note_system.notes:
        st.info("Add another note to see the cluster visualization!")
    else:
        st.info("Add some notes to see the cluster visualization!")