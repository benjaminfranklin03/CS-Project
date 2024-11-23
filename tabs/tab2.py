# ===========================================================
# Imports
# ===========================================================
import logging
import streamlit as st
from utils.graph_utils import remove_note_from_graph, rename_node_in_graph, save_graph

# ===========================================================
# Logging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# constant
# ===========================================================
GRAPH_FILE = "cache/knowledge_graph.json"

# ===========================================================
# Rendering the Note Tab (My Notes with delete, edit and show summary options)
# ===========================================================
def render_tab2(note_system):
    
    st.header("All Notes")
    if note_system.notes:
        for note_id, note in list(note_system.notes.items()):
            # unique keys for session state
            toggle_summary_key = f"show_summary_{note_id}"
            edit_mode_key = f"edit_mode_{note_id}"
            edit_title_key = f"edit_title_{note_id}"
            edit_content_key = f"edit_content_{note_id}"

            # initializing the session state variables
            if toggle_summary_key not in st.session_state:
                st.session_state[toggle_summary_key] = False
            if edit_mode_key not in st.session_state:
                st.session_state[edit_mode_key] = False

            with st.expander(f"📝 {note.title}"):
                st.write(f"**Created:** {note.created_at.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Cluster:** {note.cluster_id}")

                # Show Summary Toggle
                if st.button("Show Summary", key=f"toggle_summary_{note_id}"):
                    st.session_state[toggle_summary_key] = not st.session_state[toggle_summary_key]

                if st.session_state[toggle_summary_key]:
                    if note.summary:
                        st.write("**Summary:**")
                        st.write(note.summary)
                    else:
                        st.error("No summary available for this note.")

                # Edit Note Button
                if st.button("Edit Note", key=f"edit_note_{note_id}"):
                    st.session_state[edit_mode_key] = True

                # Editing Mode
                if st.session_state[edit_mode_key]:
                    with st.form(f"edit_form_{note_id}", clear_on_submit=False):
                        new_title = st.text_input("Edit Title", value=note.title, key=edit_title_key)
                        new_content = st.text_area("Edit Content", value=note.content, key=edit_content_key)
                        submit_edit = st.form_submit_button("Save Changes")
                        cancel_edit = st.form_submit_button("Cancel")

                    if submit_edit:
                        try:
                            old_title = note.title
                            note_system.update_note(note_id, new_content, new_title)
                            st.success(f"Note '{new_title}' updated successfully.")
                            # Update knowledge graph if title changed
                            if new_title != old_title:
                                rename_node_in_graph(st.session_state.knowledge_graph, old_title, new_title)
                                save_graph(st.session_state.knowledge_graph, GRAPH_FILE)
                            st.session_state[edit_mode_key] = False
                            st.rerun()
                            
                        except ValueError as ve:
                            st.error(str(ve))
                            logger.error(f"ValueError updating note: {ve}")
                        except Exception as e:
                            st.error("An error occurred while updating the note.")
                            logger.error(f"Exception updating note: {e}")
                    elif cancel_edit:
                        st.session_state[edit_mode_key] = False

                # Delete Note Button
                if st.button("Delete Note", key=f"delete_{note_id}"):
                    success = note_system.delete_note(note_id)
                    if success:
                        remove_note_from_graph(note.title)
                        st.success(f"Deleted note: {note.title}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete note: {note.title}")

                # Show Note Content if not in edit mode
                if not st.session_state[edit_mode_key]:
                    st.write("**Content:**")
                    st.write(note.content)
    else:
        st.info("No notes added yet. Add some notes using the sidebar!")