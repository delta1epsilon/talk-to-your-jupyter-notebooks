import os 

from langchain_community.document_loaders import NotebookLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_notebooks_docs(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.ipynb'):
            loader = NotebookLoader(
                path=os.path.join(folder_path, file_name), 
                include_outputs=True,
                max_output_length=500,
                traceback=False
            )
            docs.append(loader.load()[0])
    return docs 

def index_notebooks(folder_path):
    docs = load_notebooks_docs(folder_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    # bge-large-en models is approximatelly 1.3G
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    bge_embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=bge_embedding
    )

    retriever = vectorstore.as_retriever()
    return retriever

def get_qa_chain(retriever):
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide answer only if enough context. 
    Answer should provide a reference to the source document file name.
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = Ollama(model="llama2")

    # RAG pipeline
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return qa_chain
