import streamlit as st

from qa_rag_over_notebooks import index_notebooks, get_qa_chain


st.set_page_config(page_title="Q&A over your Jupyter Notebooks", 
                   page_icon="🧑‍💼")
st.markdown("# Q&A over your Jupyter Notebooks")

with st.sidebar:
    st.header("Index your Jupyter Notebooks")
    notebooks_folder = st.text_area('Notebooks folder:', './notebooks')
    index_button = st.button('Index')
    if index_button:
        st.session_state.retriever = index_notebooks(notebooks_folder)
        st.session_state.qa_chain = get_qa_chain(st.session_state.retriever)
        message = {
            "role": "assistant", 
            "content": f'Notebooks in {notebooks_folder} folder have been indexed'
        }
        st.session_state.messages.append(message)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Please index your Jupyter Notebooks first"}
    ]

prompt = st.chat_input("Your question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(prompt)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
