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
# Rendering the Semantic Search Tab
# ===========================================================
def render_tab4(note_system):
      
    st.header("Semantic Search")
    logger.debug("Entering Tab 4: Semantic Search")
    if note_system.notes:
        valid_notes = [note for note in note_system.notes.values() if note.embedding is not None]
        if not valid_notes:
            st.info("No notes with valid embeddings available for semantic search.")
        else:
            query = st.text_input("Enter your search query:")
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.number_input("Number of results to return (top_k):", min_value=1, max_value=20, value=5)
            with col2:
                threshold = st.slider("Minimum similarity threshold:", min_value=0.0, max_value=1.0, value=0.6)
            if query:
                if not query.strip():
                    st.warning("Query cannot be empty or whitespace!")
                else:
                    results = note_system.semantic_search(query, top_k=int(top_k), threshold=threshold)
                    if results:
                        st.subheader("Search Results:")
                        for note_id, sim_score, explanation in results:
                            note = note_system.notes[note_id]
                            with st.expander(f"📝 {note.title} (Similarity: {sim_score:.2f})"):
                                st.write(note.content)
                                st.write(f"**Explanation:** {explanation}")
                    else:
                        st.info("No matches found for your query.")
    else:
        st.info("No notes available. Add some notes to perform semantic search.")