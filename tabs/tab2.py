# ===========================================================
# Imports
# ===========================================================
import streamlit as st
from utils.graph_utils import remove_note_from_graph

# ===========================================================
# Rendering the Note Tab (My Notes with delete and show summary options)
# ===========================================================
def render_tab2(note_system):
    
    st.header("All Notes")
    if note_system.notes:
        for note_id, note in list(note_system.notes.items()):
            # Define a unique key for session state tracking
            toggle_key = f"show_summary_{note_id}"

            # Initialize the toggle state in session state if it doesn't exist
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False
            
            with st.expander(f"📝 {note.title}"):
                st.write(f"**Created:** {note.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Cluster:** {note.cluster_id}")
                
                # Toggle button for showing/hiding summary
                if st.button("Show Summary", key=f"toggle_{note_id}"):
                    st.session_state[toggle_key] = not st.session_state[toggle_key]

                # Display summary if toggled on
                if st.session_state[toggle_key]:
                    if note.summary:
                        st.write("**Summary:**")
                        st.write(note.summary)
                    else:
                        st.error("No summary available for this note.")
                
                #Delete Note button
                if st.button("Delete Note", key=f"delete_{note_id}"):
                    success = note_system.delete_note(note_id)
                    if success:
                        remove_note_from_graph(note.title)
                        st.success(f"Deleted note: {note.title}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete note: {note.title}")
                        
                # Show note content
                st.write("**Content:**")
                st.write(note.content)

    else:
        st.info("No notes added yet. Add some notes using the sidebar!")