# ===========================================================
# Imports
# ===========================================================
import streamlit as st
import plotly.express as px
import pandas as pd

# ===========================================================
# Rendering the Cluster Visualization Tab
# ===========================================================
def render_tab1(note_system):
    
    st.header("Note Clusters Visualization")
    if len(note_system.notes) >= 2:
        clusters = note_system.cluster_notes()
        coords = note_system.get_2d_visualization()

        # Prepare data for plotting
        plot_data = []
        for note_id, (x, y) in coords.items():
            note = note_system.notes[note_id]
            plot_data.append({
                'x': x,
                'y': y,
                'title': note.title,
                'cluster': f'Cluster {note.cluster_id}',
                'content': note.content[:100] + '...' if len(note.content) > 100 else note.content
            })

        df = pd.DataFrame(plot_data)

        # Create scatter plot
        fig = px.scatter(
            df, x='x', y='y', color='cluster',
            hover_data=['title', 'content'],
            title='Note Clusters Visualization'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display cluster information
        st.subheader("Cluster Information")
        for cluster_id, note_ids in clusters.items():
            with st.expander(f"Cluster {cluster_id} ({len(note_ids)} notes)"):
                for note_id in note_ids:
                    note = note_system.notes[note_id]
                    st.write(f"**{note.title}**")
                    st.write(note.content[:200] + '...' if len(note.content) > 200 else note.content)
    elif note_system.notes:
        st.info("Add another note to see the cluster visualization!")
    else:
        st.info("Add some notes to see the cluster visualization!")