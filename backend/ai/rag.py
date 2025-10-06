import os
import numpy as np
import psycopg2
import logging
import json
from dotenv import load_dotenv
from typing import List, Dict, Union, Any, Set, Tuple

# LangChain Imports
from langchain_cerebras import ChatCerebras
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (assuming your .env file is present)
load_dotenv()

# ==============================================================================
# 1. Configuration and MOCKED Embedding Client
# ==============================================================================

AZURE_CONFIG = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY")
}

class AzureEmbeddingClient:
    """Simulated Azure OpenAI Client for generating embeddings."""
    def __init__(self, config: dict):
        self.config = config

    def get_embedding(self, text: str) -> List[float]:
        """
        Mocks the call to the Azure OpenAI embedding model (1536 dimensions).
        """
        # Generate a reproducible dummy vector for testing (1536 dimensions)
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(1536).tolist()

# Global Embedder Instance
AZURE_EMBEDDER = AzureEmbeddingClient(AZURE_CONFIG)

# ==============================================================================
# 2. Cerebras LLM Setup and Query Expansion
# ==============================================================================

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GLOBAL_LLM: Union[ChatCerebras, None] = None

if CEREBRAS_API_KEY:
    try:
        GLOBAL_LLM = ChatCerebras(
            model="llama-3.3-70b", 
            api_key=CEREBRAS_API_KEY,
            temperature=0.7,
            max_tokens=2048 # Increased max_tokens for Reranking Step
        )
    except Exception as e:
        logger.error(f"Failed to initialize Cerebras LLM: {e}")
else:
    logger.warning("CEREBRAS_API_KEY not set. LLM-based functions will be mocked.")


def expand_query(query: str) -> List[str]:
    """
    Expands a single query into 5 related queries (including the original) 
    using the Cerebras LLM.
    """
    if not GLOBAL_LLM:
        logger.warning("Expansion LLM not available. Returning original query only.")
        return [query]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a helpful query rewriter. Your task is to rephrase the user's "
             "input query into 4 distinct, semantically similar variations to improve "
             "search recall. Respond ONLY with a numbered list of the 4 new queries. "
             "Do not include the original query, any introductory text, or explanations."
            ),
            ("human", f"Original Query: {query}"),
        ]
    )

    try:
        chain = prompt | GLOBAL_LLM
        response = chain.invoke({"query": query}).content
        
        # Parse the LLM response
        new_queries = []
        for line in response.split('\n'):
            if line and (line[0].isdigit() or line.startswith('-')):
                parsed_query = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                if parsed_query:
                    new_queries.append(parsed_query)

        # Combine the original query with the 4 new queries
        return [query] + new_queries[:4]
    
    except Exception as e:
        logger.error(f"Error during query expansion: {e}")
        return [query] # Fallback to original query


# ==============================================================================
# 3. LLM Reranking Function
# ==============================================================================

def llm_rerank_chunks(query: str, documents: Dict[str, str], top_k: int = 10) -> List[str]:
    """
    Uses the LLM to rerank the retrieved documents based on relevance to the query 
    and returns the top_k chunk texts.
    
    Args:
        query: The original user question.
        documents: A dictionary mapping a unique ID (e.g., "DOC_1") to the chunk text.
        top_k: The number of best documents to return.
    
    Returns:
        A list of the top_k most relevant chunk texts.
    """
    if not GLOBAL_LLM:
        logger.warning("Reranking LLM not available. Returning first 10 documents arbitrarily.")
        return list(documents.values())[:top_k]

    # 1. Format Documents for the LLM
    formatted_docs = "\n\n".join([
        f"--- DOCUMENT ID: {doc_id} ---\n{text}"
        for doc_id, text in documents.items()
    ])

    # 2. Construct the Reranking Prompt
    reranking_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are an expert relevance ranking engine. Your task is to analyze the "
             "provided documents and select the {top_k} documents that are MOST relevant "
             "to the user's query. You MUST respond ONLY with a comma-separated list of "
             "the selected DOCUMENT IDs (e.g., 'DOC_3, DOC_8, DOC_1'). Do not include "
             "any other text, explanation, or markdown formatting."
            ),
            ("human", 
             "Original Query: {query}\n\n"
             "--- DOCUMENTS FOR RANKING ---\n"
             "{documents}"
             "\n\nBased on the query, identify the {top_k} best DOCUMENT IDs."
            ),
        ]
    )

    try:
        chain = reranking_prompt | GLOBAL_LLM
        
        response = chain.invoke({
            "query": query, 
            "documents": formatted_docs,
            "top_k": top_k
        }).content
        
        # 3. Parse the LLM Response
        selected_ids = [
            doc_id.strip() for doc_id in response.split(',') 
            if doc_id.strip().startswith("DOC_")
        ][:top_k]

        # 4. Map IDs back to Texts
        reranked_texts = [
            documents[doc_id] for doc_id in selected_ids if doc_id in documents
        ]
        
        # Ensure we return exactly top_k if available, or fewer if the LLM failed to parse correctly
        return reranked_texts[:top_k]
        
    except Exception as e:
        logger.error(f"Error during LLM reranking: {e}")
        return list(documents.values())[:top_k] # Fallback

# ==============================================================================
# 4. PostgreSQL Configuration and Core Search Function
# ==============================================================================

DB_PARAMS: Dict[str, str] = {
    'dbname': os.getenv('PGDATABASE'),
    'user': os.getenv('PGUSER'),
    'password': os.getenv('PGPASSWORD'),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '5432')
}

def get_top_k_chunks(
    query_vector_str: str, 
    k: int
) -> List[str]:
    """
    Performs a single vector search, returning a list of unique chunk texts.
    """
    unique_texts = [] # Use a list here to preserve order of relevance for the search

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Select chunk_text and order by cosine distance
        sql_query = f"""
        SELECT 
            c.chunk_text
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        ORDER BY c.embedding <=> '{query_vector_str}' ASC
        LIMIT {k};
        """

        cur.execute(sql_query)
        fetched_results = cur.fetchall()
        
        cur.close()
        conn.close()

        # Extract only the text
        for (chunk_text,) in fetched_results:
            unique_texts.append(chunk_text)

        return unique_texts

    except Exception as e:
        logger.error(f"PostgreSQL Search Function Error: {e}")
        return []

# ==============================================================================
# 5. Master RAG Function
# ==============================================================================

def rag_vector_search_with_reranking(
    query_text: str, 
    k_per_query: int = 10,
    final_top_k: int = 10
) -> List[str]:
    """
    Master function: Expands query, searches, de-duplicates (up to 50), 
    and uses LLM for final top_k reranking.
    """
    
    k_per_query = max(1, min(10, k_per_query)) 
    final_top_k = max(1, min(50, final_top_k)) # Cap final results at 50, but usually 10

    # 1. Expand the query (gets up to 5 queries)
    expanded_queries = expand_query(query_text)
    
    # Use a dictionary to store unique texts and map them to an ID for the reranker
    # Key: Chunk Text (str) | Value: Unique ID (str)
    all_unique_texts: Dict[str, str] = {}
    
    print(f"--- Running RAG Search (Expanded Queries: {len(expanded_queries)}) ---")
    
    # 2. Perform searches and aggregate unique texts
    for i, query in enumerate(expanded_queries):
        print(f"Searching with query {i+1}: '{query[:50]}...'")

        query_embedding = AZURE_EMBEDDER.get_embedding(query)
        vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Get top K results (text only)
        current_texts = get_top_k_chunks(vector_str, k=k_per_query)
        
        # Add texts to the unique dictionary. We assign a temporary ID for reranking.
        for text in current_texts:
            if text not in all_unique_texts:
                doc_id = f"DOC_{len(all_unique_texts) + 1}"
                all_unique_texts[text] = doc_id

        # Stop if we hit 50 total documents to prevent hitting LLM context limits
        if len(all_unique_texts) >= 50:
            print("Reached 50 unique documents. Halting search.")
            break
                
    # 3. Prepare the documents for Reranking (mapping ID -> Text)
    # The keys and values are swapped for easier LLM ID lookup later
    docs_for_rerank = {v: k for k, v in all_unique_texts.items()}
    
    print(f"\n--- Total Unique Documents Fetched: {len(docs_for_rerank)} ---")
    
    # 4. LLM Reranking
    reranked_final_texts = llm_rerank_chunks(
        query=query_text, 
        documents=docs_for_rerank, 
        top_k=final_top_k
    )
                
    # 5. Return the final list of top K texts
    return reranked_final_texts

# ==============================================================================
# 6. Example Usage (for testing)
# ==============================================================================

if __name__ == "__main__":
    search_query = "What is the primary function of the James Webb Space Telescope's MIRI instrument?"
    
    # Perform the search: 5 queries * 10 chunks each, reranked to 10 final chunks
    top_chunks_texts = rag_vector_search_with_reranking(search_query, k_per_query=10, final_top_k=10)

    if top_chunks_texts:
        print(f"\nSuccessfully retrieved {len(top_chunks_texts)} FINAL RERANKED chunks:")
        
        for i, text in enumerate(top_chunks_texts):
            print(f"--- Reranked Result {i+1} ---")
            print(f"Text: {text[:200]}...\n")
        
    else:
        print("\nNo results found or an error occurred during search/reranking.")