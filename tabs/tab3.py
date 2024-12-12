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
        
    st.header("Find Similar Notes")
    logger.debug("Entering Tab 3: Find Similar Notes")
    
    # check if there are enough notes for comparison
    if len(note_system.notes) >= 2:
        valid_notes = [note for note in note_system.notes.values() if note.embedding is not None]
        
        if len(valid_notes) < 2:
            st.info("Not enough notes with valid embeddings to find similar notes.")
            
        else:
            note_titles = {note.title: note_id for note_id, note in note_system.notes.items()}
            selected_title = st.selectbox("Select a note to find similar ones:", list(note_titles.keys()))
            
            # get similar notes if a note is selected
            if selected_title:
                selected_id = note_titles[selected_title]
                similar_notes = note_system.get_similar_notes(selected_id)
                
                if similar_notes:
                    st.subheader("Similar Notes:")
                    
                    for note_id, similarity in similar_notes:
                        note = note_system.notes[note_id]
                        toggle_summary_key = f"tab3_show_summary_{note_id}"  # prefix to make it different from tab2
                        
                        # initialize toggle state for summary display
                        if toggle_summary_key not in st.session_state:
                            st.session_state[toggle_summary_key] = False
                        
                        with st.expander(f"📝 {note.title} (Similarity: {similarity:.2f})"):
                            st.write("\n")
                            
                            # toggle summary button
                            if st.button("Show Summary", key=f"tab3_toggle_summary_{note_id}"):  # Prefix to ensure uniqueness
                                st.session_state[toggle_summary_key] = not st.session_state[toggle_summary_key]

                            # display summary if toggled on
                            if st.session_state[toggle_summary_key]:
                                if note.summary:
                                    st.write("**Summary:**")
                                    st.write(note.summary)
                                    
                                else:
                                    st.error("No summary available for this note.")
                                    
                            # display content
                            st.write("\n")
                            st.write("**Content:**")
                            st.write(note.content)
                            
                else:
                    st.info("No similar notes found. Add more notes to find similarities.")
                    
    elif note_system.notes:
        st.info("Add another note to find similar notes!")
        
    else:
        st.info("Add some notes to find similarities!")