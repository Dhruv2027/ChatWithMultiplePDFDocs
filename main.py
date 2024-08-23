import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Get PDFs (Rather, any file)
def getPDFText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Convert the text from the file into smaller chunks
def make_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 100)
    chunks = text_splitter.split_text(text)
    return chunks

#Convert the chunks into embeddings
def get_vectorStore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    PROMPT_TEMPLATE = """
    Answer the question with as much detail as possible from the provided context.
    If the answer is not in the provided context, say: "answer is not available in context" dont't provide the wrong answer.
    Context: \n {context} \n
    Question: \n {question} \n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = PROMPT_TEMPLATE, input_variables = ["context", "question"])

    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)
    return chain

def userInput(question):
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embedding,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question":question},
        return_only_outputs = True
    )
    print(response)
    st.write("reply:", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        userInput(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = getPDFText(pdf_docs)
                text_chunks = make_chunks(raw_text)
                get_vectorStore(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()