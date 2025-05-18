!pip install streamlit langchain langchain-community openai pinecone-client pypdf
from langchain_community.document_loaders import PyPDFLoader

# Specify the path to your PDF file
file_path = r"C:\Users\Admin\Downloads\attentionn.pdf"

# Initialize the loader
loader = PyPDFLoader(file_path)

# Load and split the document into chunks
documents = loader.load_and_split()

# Check the number of chunks
print(f"Number of chunks: {len(documents)}")
import openai
from langchain.embeddings import OpenAIEmbeddings
import pinecone

# Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings()

# Generate embeddings for each document chunk
embedded_docs = embeddings.embed_documents([doc.page_content for doc in documents])

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")

# Create or connect to a Pinecone index
index = pinecone.Index("document-index")

# Prepare vectors for upsertion
vectors = [
    {
        "id": str(i),
        "values": embedded_docs[i],
        "metadata": {"source": documents[i].metadata["source"]}
    }
    for i in range(len(documents))
]

# Upsert vectors into the index
index.upsert(vectors=vectors)
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize the vector store
vectorstore = Pinecone(index, embeddings.embed_query)

# Initialize the language model
llm = OpenAI(model="text-davinci-003")

# Set up the retrieval-based QA chain
qa_chain = RetrievalQA(combine_docs_chain=llm, retriever=vectorstore.as_retriever())

# Streamlit interface
st.title("Document Q&A Chatbot with Memory")

user_query = st.text_input("Ask a question:")

if user_query:
    # Retrieve answer
    answer = qa_chain.run(user_query)

    # Store the conversation in memory
    st.session_state.memory.add_user_message(user_query)
    st.session_state.memory.add_ai_message(answer)

    # Display the answer
    st.write(answer)
