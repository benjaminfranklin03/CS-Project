# ===========================================================
# Imports
# ===========================================================
import os
import logging
import streamlit as st
from openai import OpenAI

# ===========================================================
# Loggging
# ===========================================================
logger = logging.getLogger(__name__)

# ===========================================================
# Adjustable variables
# ===========================================================
model = 'gpt-4o'
k = 3 #number of notes to retrieve

# ===========================================================
# Helper function to get response (for caching data)
# ===========================================================
@st.cache_data #caching response
def get_response(prompt, model):
    # Initializing OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # Send the prompt to the LLM via the OpenAI API
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o"
        )

        # Access the generated reply
        answer = response.choices[0].message.content.strip()
        # Display the answer
        st.subheader("Answer:")
        st.write(answer)
                
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        logger.error(f"Error during LLM API call: {e}")
    return answer

# ===========================================================
#  Q&A with RAG
# ===========================================================
def render_tab6(notesystem):
    
    st.header("Q&A")
    # Input field
    question = st.text_input("Enter your question:")
    # Perform semantic search to retrieve the top-k most relevant notes
    if question:
        results = notesystem.semantic_search(query=question, top_k=k)

        #Once the results are found, make the API call
        if results:
            # Construct the context from retrieved notes
            retrieved_info = ""
            for note_id, sim_score, explanation in results:
                note = notesystem.notes[note_id]
                # Append note title and summary to the retrieved information
                retrieved_info += f"Topic: {note.title}\nSummary: {note.summary}\n\n"
                
            # Construct the prompt
            prompt = (
                f"Question: {question}\n\n"
                f"Based on the following retrieved summaries, provide a detailed response:\n\n"
                f"{retrieved_info}"
            )
            # Make the API call
            get_response(prompt, model)
            
        else:
            st.info("No relevant notes found for your query.")