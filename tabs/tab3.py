# ===========================================================
# Imports
# ===========================================================
import streamlit as st
import logging

# ===========================================================
# Logging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# Rendering the Similar Notes Tab
# ===========================================================
def render_tab3(note_system):
    """Render the Similar Notes Tab."""    
    st.header("Find Similar Notes")
    logger.debug("Entering Tab 3: Find Similar Notes")
    if len(note_system.notes) >= 2:
        valid_notes = [note for note in note_system.notes.values() if note.embedding is not None]
        if len(valid_notes) < 2:
            st.info("Not enough notes with valid embeddings to find similar notes.")
        else:
            note_titles = {note.title: note_id for note_id, note in note_system.notes.items()}
            selected_title = st.selectbox("Select a note to find similar ones:", list(note_titles.keys()))
            if selected_title:
                selected_id = note_titles[selected_title]
                similar_notes = note_system.get_similar_notes(selected_id)
                if similar_notes:
                    st.subheader("Similar Notes:")
                    for note_id, similarity in similar_notes:
                        note = note_system.notes[note_id]
                        with st.expander(f"📝 {note.title} (Similarity: {similarity:.2f})"):
                            st.write(note.content)
                else:
                    st.info("No similar notes found. Add more notes to find similarities.")
    elif note_system.notes:
        st.info("Add another note to find similar notes!")
    else:
        st.info("Add some notes to find similarities!")