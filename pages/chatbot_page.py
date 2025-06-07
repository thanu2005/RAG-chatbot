import streamlit as st
from genai_services import answer_with_context
from chroma_services import query_documents

st.title("RAG QnA Chatbot")
st.write("Ask questions about your ingested document!")

# Chat input for user's question
user_query = st.chat_input("Your question:")

if user_query:
    # Query Chroma for context documents
    context_chunks = query_documents(user_query, n_results=3)

    with st.spinner("Generating answer..."):
        # Generate answer using LLM with retrieved context
        answer = answer_with_context(user_query, context_chunks)

        # Display the answer
        st.markdown(f"**Answer:** {answer}")

        # Optional: Show retrieved context for transparency/debugging
        with st.expander("Show retrieved context"):
            st.write("\n".join(context_chunks))
