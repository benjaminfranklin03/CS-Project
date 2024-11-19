# ===========================================================
# Imports
# ===========================================================
import streamlit as st
from utils.graph_utils import remove_note_from_graph

# ===========================================================
# Rendering the Note Tab
# ===========================================================
def render_tab2(note_system):
    
    st.header("All Notes")
    if note_system.notes:
        for note_id, note in list(note_system.notes.items()):
            with st.expander(f"📝 {note.title}"):
                st.write(f"**Created:** {note.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Cluster:** {note.cluster_id}")
                st.write("**Content:**")
                st.write(note.content)
                if st.button("Delete Note", key=f"delete_{note_id}"):
                    success = note_system.delete_note(note_id)
                    if success:
                        remove_note_from_graph(note.title)
                        st.success(f"Deleted note: {note.title}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete note: {note.title}")
    else:
        st.info("No notes added yet. Add some notes using the sidebar!")