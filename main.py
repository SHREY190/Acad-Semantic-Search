import streamlit as st
from views import ask_question

def main():
    st.title("Semantic Search Engine")

    # User input
    query = st.text_input("Ask a question")

    # If user has entered a query
    if query:
        # Get answer from GPT-3
        answer = ask_question(query)
        
        # Display answer
        st.markdown(f"**{query}**\n\n{answer}")

if __name__ == "__main__":
    main()