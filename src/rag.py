import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =====================================================================
# PHASE 1: Document Loading, Chunking, & Vector Storage (ChromaDB)
# =====================================================================
doc_path = os.path.join("docs", "knowledge_base.txt")

# 1. Load the document from disk
loader = TextLoader(doc_path)
documents = loader.load()

# 2. Split text into manageable chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 3. Create local vector embeddings & save them to ChromaDB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# ===================================================================== 
# # PHASE 2: RAG Retrieval &amp; Answer Generation Function # #
# =====================================================================

# 4. Instantiate ChromaDB Retriever and Llama 3.2 LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = OllamaLLM(model="llama3.2")



def get_rag_response(user_query: str) -> dict:
    """
    1. Fetches relevant document chunks from ChromaDB.
    2. Passes the chunks + question to Llama 3.2 via LCEL chain.
    3. Returns a dictionary containing answer and contexts for DeepEval.
    """
    # Step A: Retrieve relevant context chunks from ChromaDB
    retrieved_docs = retriever.invoke(user_query)
    
    # Step B: Combine chunk text contents into a single string context
    context_text = "nn".join(doc.page_content for doc in retrieved_docs)
    
    # Step C: Prompt template enforcing grounding in context
    prompt = ChatPromptTemplate.from_template(""" You are a helpful assistant. Answer the user's question thoroughly using strictly the provided context. 
    Make sure to address all specific conditions, edge cases, or policies mentioned in the question. 
    Context: {context} Question: {question} Answer: """)
    
    # Step D: Construct the LangChain LCEL pipeline (Prompt -> LLM -> Parser)
    chain = prompt | llm | StrOutputParser()
    
    # Step E: Invoke the LLM with context and question
    answer = chain.invoke({"context": context_text, "question": user_query})
    
    # Step F: Return dictionary formatted for DeepEval test cases
    return {
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs]
    }

# =====================================================================
# Step 3: Verification / Local Execution Block
# =====================================================================
if __name__ == "__main__":
    test_question = "What credit does an Enterprise customer get if uptime drops below 99.0%?"
    
    print(f"Testing RAG Pipeline with Question: '{test_question}'n")
    
    result = get_rag_response(test_question)
    
    print("--- Generated Answer ---")
    print(result["answer"])
    
    print("n--- Retrieved Context Chunks ---")
    for idx, ctx in enumerate(result["retrieved_contexts"], 1):
        print(f"Chunk {idx}:n{ctx}n")
