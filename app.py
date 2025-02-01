import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from the file
load_dotenv()
# Access the API key
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#pdf data is stored in 'data' in list form
loader=PyPDFDirectoryLoader("pdfs")
data = loader.load() 

#tokenisation - divides data into tokens (list form)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)      #each chunk 500 characters long 20 characters of one chunk overlaps with the next chunk
text_chunks = text_splitter.split_documents(data)

#creating an object of OpenAIEmbeddings
embedding = OpenAIEmbeddings()

#embedding vectors in pinecone
import os
from pinecone import Pinecone, ServerlessSpec

api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

#initializing index name
index_name = "testing"

#CREATE EMBEDDINGS FOR EACH TEXT CHUNK
from langchain.vectorstores import Pinecone as LangchainPinecone # Import Pinecone from langchain.vectorstores

docsearch = LangchainPinecone.from_texts([t.page_content for t in text_chunks], embedding, index_name=index_name) # Use LangchainPinecone instead of Pinecone

# Create a RetrievalQA chain using the Pinecone index and an OpenAI language model
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

#UI portion 
def main():
    st.title("PDF Query System")
    st.write("Ask a question based on the uploaded PDFs.")
    
    # User input for query
    user_input = st.text_input("Enter your query:", "")
    
    if st.button("Search"):
        if user_input.strip():
            with st.spinner("Fetching answer..."):
                result = qa({'query': user_input})
                st.success("Answer:")
                st.write(result['result'])
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
