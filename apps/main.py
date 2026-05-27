from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from fastapi import FastAPI
from dotenv import load_dotenv
from apps.rag_pipeline import rag_pipeline
from pydantic import BaseModel

import os
 # Load environment variables from .env file
 # load the document 
load_dotenv()
query="What is the leave policy of the company?"
loader=PyPDFLoader("data/hr_policy.pdf")
docs=loader.load()
print("document loaded successfully ✅")
# splitting documents
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunk=text_splitter.split_documents(docs)
print("documeny splitted successfully ✅")
# convert into embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("embeddings created successfully ✅")

vectorestore=FAISS.from_documents(chunk,embeddings)
vectorestore.save_local("faiss_index")
print("faiss")
bm25=BM25Retriever.from_documents(chunk)
bm25.k=5
print("BM25 retriever created successfully ✅")
# combine both retrievers
faiss_docs=vectorestore.similarity_search(query,k=5)
bm2_docs=bm25.invoke(query)

all_retrieved_docs=faiss_docs+bm2_docs
print("combined retriever created successfully ✅")

# Reranking process
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs=[[query,docs.page_content] for docs in all_retrieved_docs]
score=reranker.predict(pairs)
print("score:",score)

# sort the documents based on the scores
ranked=sorted(zip(score,all_retrieved_docs),reverse=True)
context = "\n\n".join([doc.page_content for _, doc in ranked])
print("context created successfully ✅")

prompt=PromptTemplate(
    input_variables=["context","question"],
    template=""" You are an enterprise AI assistant.

Use only the provided context.

Context:
{context}

Question:
{question}

Answer:"""
)

print(prompt)
print("prompt created successfully ✅")

load_dotenv()
groq_api_key =os.getenv("GROQ_API_KEY")
llm=ChatGroq(model_name="llama-3.3-70b-versatile",
             temperature=0.2,
             api_key=groq_api_key)
print("LLM loaded successfully ✅")

final_prompt=prompt.format(
    context= context,
    question=query
)
print(final_prompt)

print("prompt :",final_prompt)
print("prompt formatted successfully ✅")

response = llm.invoke(final_prompt)
print(response)
print ("success")

#citation
for doc in all_retrieved_docs:
    print(doc.metadata)

sources=[doc.metadata["source"] 
         for do in all_retrieved_docs
         ]

print("sources extracted successfully ✅")

# FastAPI app
app= FastAPI()
class Query_request(BaseModel):
    query: str
@app.post("/chat")
async def chat(query:str):
    response=rag_pipeline(query)
    return {"response":response}