import os
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_resume(query: str) -> str:
    qdrant_client = QdrantClient(host="localhost", port=6333)

    try:
        qdrant_client.get_collection(collection_name="resume_rag")
        print("Using existing collection: resume_rag")
    except Exception as e:
        print(f"Creating new collection: resume_rag (Error: {str(e)})")
        qdrant_client.create_collection(
            collection_name="resume_rag",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE), 
        )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name="resume_rag",
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    response = qa_chain.invoke({"query": query})["result"]
    return response