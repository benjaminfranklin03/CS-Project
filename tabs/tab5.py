# ===========================================================
# Imports
# ===========================================================
import os
import logging
import streamlit as st
from utils.note_utils import NoteEmbeddingSystem
from openai import OpenAI  

# ===========================================================
# Logging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# Adjustable Variables
# ===========================================================
MODEL = 'gpt-4o'  
TOP_K = 3         # Number of notes to retrieve for RAG

# ===========================================================
# Helper Functions
# ===========================================================
@st.cache_data
def generate_model_response(prompt, model):
    """
    Generate a response from the language model using the given prompt.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        st.error("An error occurred while contacting the language model.")
        return ""

def construct_rag_prompt(question, last_response, retrieved_notes):
    """
    Construct a RAG prompt using only the last model output and newly retrieved notes.
    """
    # Include last model output
    context = f"Model: {last_response}" if last_response else ""

    # Add newly retrieved notes
    retrieved_info = "\n\n".join(
        f"Note: {note['title']}\nSummary: {note['summary']}" for note in retrieved_notes
    )

    # Combine all parts to form the prompt
    prompt = (
        f"Context:\n{context}\n\n"
        f"Retrieved Notes:\n{retrieved_info}\n\n"
        f"User: {question}\n"
        f"Please format all math expressions using LaTeX, enclosed with $ for inline math and $$ for block math.\n"
        f"Model:"
    )
    return prompt

# ===========================================================
# Tab 6: Q&A with Follow-Up
# ===========================================================
def render_tab5(note_system: NoteEmbeddingSystem):
    """
    Render the Q&A tab with follow-up support using optimized RAG.
    """
    st.header("Q&A with Follow-Up")

    # Initialize last response and history in session state
    if 'last_model_response' not in st.session_state:
        st.session_state['last_model_response'] = ""
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    # === Form for Question Input === (so we can clear the note history without reloading the query)
    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Enter your question:")
        submit_button = st.form_submit_button("Submit")

    if submit_button and question:
        # Perform semantic search to retrieve relevant notes
        search_results = note_system.semantic_search(query=question, top_k=TOP_K)

        # Extract retrieved notes
        retrieved_notes = [
            {'title': note_system.notes[note_id].title, 'summary': note_system.notes[note_id].summary}
            for note_id, _ in search_results
        ]

        # Construct the prompt for RAG
        prompt = construct_rag_prompt(question, st.session_state['last_model_response'], retrieved_notes)

        # Generate response from the model
        response = generate_model_response(prompt, MODEL)

        # Update last model response and conversation history
        st.session_state['last_model_response'] = response
        st.session_state['conversation_history'].append({
            'question': question,
            'retrieved_notes': retrieved_notes,
            'response': response
        })

        # Display model response in markdown to support LaTeX
        st.markdown(f"**Question:** {question}")
        st.subheader("Model's Response:")
        st.markdown(response)

    # === Conversation History and Clear Button ===
    if st.session_state['conversation_history']:
        st.subheader("Conversation History")

        # Add "Clear History" button under the header
        clear_button = st.button("Clear History")
        if clear_button:
            # Clear conversation-related session state
            st.session_state['conversation_history'] = []
            st.session_state['last_model_response'] = ""
            st.success("Conversation history cleared!")

        # Display the history
        for idx, item in enumerate(st.session_state['conversation_history'][:-1], 1):
            st.write(f"**{idx}. User:** {item['question']}")
            st.markdown(f"**Model:** {item['response']}")
            if item['retrieved_notes']:
                with st.expander(f"Relevant Notes for Question {idx}"):
                    for note in item['retrieved_notes']:
                        st.markdown(f"**{note['title']}**")
                        st.markdown(note['summary'])
