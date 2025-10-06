from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import os
import datetime
from dotenv import load_dotenv
import psycopg2
import numpy as np
from typing import List, Dict, Any, Union, Set, Tuple
from langchain_cerebras import ChatCerebras
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ==============================================================================
# 1. CONFIGURATION & MOCKED EMBEDDING CLIENT
# ==============================================================================

AZURE_CONFIG = {
    "endpoint": os.getenv("AZURE_API_ENDPOINT"),
    "api_version": os.getenv("AZURE_API_VERSION", "2024-02-01"),
    "api_key": os.getenv("AZURE_EMBEDDINGS_API_KEY")
}

class AzureEmbeddingClient:
    """Simulated Azure OpenAI Client for generating embeddings."""
    def __init__(self, config: dict):
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self):
        if not all([self.config.get(k) for k in ["endpoint", "api_key"]]):
            logger.warning("Azure OpenAI config incomplete. Using dummy vectors for embeddings.")
            return "AzureOpenAI Client (MOCKED)"
        return "AzureOpenAI Client (MOCKED)" 

    def get_embedding(self, text: str) -> List[float]:
        """Mocks the call to the Azure OpenAI embedding model (1536 dimensions)."""
        if self.client == "AzureOpenAI Client (MOCKED)":
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(1536).tolist()
        return []

try:
    AZURE_EMBEDDER = AzureEmbeddingClient(AZURE_CONFIG)
except ValueError as e:
    logger.error(f"Azure Embedder initialization failed: {e}. Tool will use dummy vectors.")
    AZURE_EMBEDDER = None

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

DB_PARAMS: Dict[str, str] = {
    'dbname': os.getenv('PGDATABASE'),
    'user': os.getenv('PGUSER'),
    'password': os.getenv('PGPASSWORD'),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '5432')
}

# ==============================================================================
# 2. ADVANCED RAG TOOL (Core Retrieval Logic)
# ==============================================================================

class AdvancedRAGInput(BaseModel):
    """Input schema for the Advanced RAG Tool."""
    query: str = Field(description="The user's question or search phrase for the knowledge base.")

class AdvancedRAGTool(BaseTool):
    """
    A unified RAG tool performing Query Expansion, Multi-Query Vector Search, 
    and LLM Re-ranking to retrieve the top 10 most relevant chunks.
    """
    name: str = "advanced_rag_search"
    description: str = "Retrieves and re-ranks context for the RAG chain."
    args_schema: type[BaseModel] = AdvancedRAGInput
    
    llm: ChatCerebras
    
    def _get_embedding(self, query: str) -> List[float]:
        if AZURE_EMBEDDER:
            return AZURE_EMBEDDER.get_embedding(query)
        np.random.seed(hash(query) % 2**32)
        return np.random.rand(1536).tolist()

    def _expand_query(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful query rewriter. Rephrase the user's input query "
                       "into 4 distinct, semantically similar variations. Respond ONLY with a "
                       "numbered list of the 4 new queries. Do not include the original query."),
            ("human", f"Original Query: {query}"),
        ])
        try:
            chain = prompt | self.llm
            response = chain.invoke({"query": query}).content
            new_queries = []
            for line in response.split('\n'):
                if line and (line[0].isdigit() or line.startswith('-')):
                    parsed_query = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                    if parsed_query:
                        new_queries.append(parsed_query)
            return [query] + new_queries[:4]
        except Exception as e:
            logger.error(f"Tool Query Expansion Error: {e}")
            return [query]

    def _vector_search_db(self, query_vector_str: str, k: int) -> List[Tuple[str, int]]:
        """
        Performs a single vector search, returning a list of tuples: 
        (chunk_text, article_id)
        """
        unique_texts_and_ids = []
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            cur = conn.cursor()
            sql_query = f"""
            SELECT c.chunk_text, c.article_id
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            ORDER BY c.embedding <=> '{query_vector_str}' ASC
            LIMIT {k};
            """
            cur.execute(sql_query)
            fetched_results = cur.fetchall()
            cur.close()
            conn.close()

            for chunk_text, article_id in fetched_results:
                unique_texts_and_ids.append((chunk_text, article_id))
            return unique_texts_and_ids
        except Exception as e:
            logger.error(f"Tool DB Search Error: {e}")
            return []

    def _llm_rerank_chunks(self, query: str, documents: Dict[str, str], top_k: int) -> List[str]:
        if not documents: return []
        formatted_docs = "\n\n".join([f"--- DOCUMENT ID: {doc_id} ---\n{text}" for doc_id, text in documents.items()])

        reranking_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert relevance ranking engine. Select the {top_k} documents "
                       "that are MOST relevant to the user's query. You MUST respond ONLY with a "
                       "comma-separated list of the selected DOCUMENT IDs (e.g., 'DOC_3, DOC_8'). "
                       "Do not include any other text."),
            ("human", "Original Query: {query}\n\n--- DOCUMENTS FOR RANKING ---\n"
                      "{documents}\n\nIdentify the {top_k} best DOCUMENT IDs."),
        ])

        try:
            chain = reranking_prompt | self.llm
            response = chain.invoke({"query": query, "documents": formatted_docs, "top_k": top_k}).content
            
            selected_ids = [doc_id.strip() for doc_id in response.split(',') if doc_id.strip().startswith("DOC_")][:top_k]
            return [documents[doc_id] for doc_id in selected_ids if doc_id in documents]
        except Exception as e:
            logger.error(f"Tool LLM Reranking Error: {e}")
            return list(documents.values())[:top_k]

    # --- Tool RUNNER (Returns Formatted String of Top 10 Chunks) ---

    def _run(self, query: str) -> str:
        """
        Executes the full RAG pipeline: Expand -> Search -> Rerank (Chunks) 
        -> Returns a single string containing the top 10 chunks.
        """
        
        K_PER_QUERY = 10
        FINAL_TOP_K = 10
        MAX_DOCS = 50

        # Key: Chunk Text | Value: (Unique DOC ID, article_id)
        all_unique_texts: Dict[str, Tuple[str, int]] = {} 
        
        try:
            # 1. Query Expansion
            expanded_queries = self._expand_query(query)

            # 2. Multi-Query Search and Aggregation
            for q in expanded_queries:
                if len(all_unique_texts) >= MAX_DOCS: break
                
                query_embedding = self._get_embedding(q)
                vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

                # current_results is List[Tuple[str, int]] -> (chunk_text, article_id)
                current_results = self._vector_search_db(vector_str, k=K_PER_QUERY)
                
                for text, article_id in current_results:
                    if text not in all_unique_texts:
                        doc_id = f"DOC_{len(all_unique_texts) + 1}"
                        all_unique_texts[text] = (doc_id, article_id)
            
            if not all_unique_texts:
                return "No relevant context was found in the knowledge base after expansion and search."

            # 3. Prepare for Chunk Reranking (Map ID -> Text)
            docs_for_rerank = {v[0]: k for k, v in all_unique_texts.items()}
            
            # 4. LLM Reranking: Get the texts of the top 10 chunks
            reranked_chunk_texts = self._llm_rerank_chunks(
                query=query, 
                documents=docs_for_rerank, 
                top_k=FINAL_TOP_K
            )
            
            if not reranked_chunk_texts:
                 return "The search retrieved context, but the LLM reranking step failed to select the top documents."

            # 5. Format the final output string (low token count)
            formatted_output = "Retrieved and Reranked Context Chunks (Top 10):\n"
            for i, chunk_text in enumerate(reranked_chunk_texts):
                formatted_output += f"--- Chunk {i+1} ---\n{chunk_text}\n"

            return formatted_output

        except Exception as e:
            logger.error(f"Advanced RAG Tool Error (Critical): {e}", exc_info=True)
            return f"Error: Critical RAG failure. Details: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

# ==============================================================================
# 3. SYNTHESIS AND CLASSIFIER CHAIN SETUP
# ==============================================================================

# Synthesis Prompt Template (Used for both RAG and Conversational answers)
SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            "You are a specialized AI assistant focused on **space biology, aerospace medicine, "
            "and life science research in microgravity**. Your goal is to provide a concise, accurate, "
            "and conversational answer to the user's question, using the language of a knowledgeable researcher. "
            "If context is provided, your answer MUST be based ONLY on that context. If the context is "
            "'No external context required.', answer the user's question conversationally based on your "
            "general knowledge of space biology."
        )
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Context:\n---\n{context}\n---\n\nQuestion: {input}"),
])

# New: Classification Prompt for determining RAG necessity
CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Analyze the user's question. If it is conversational, generic (e.g., 'Hello', 'How are you?'), "
               "or requires general knowledge NOT specific to **space biology or related aerospace research**, respond ONLY with 'NO'. "
               "If it requires specific factual or external knowledge related to **space biology, microgravity research, or astronaut health**, "
               "respond ONLY with 'YES'."),
    ("user", "{input}"),
])


# --- Global Initialization ---
GLOBAL_LLM: Union[ChatCerebras, None] = None
GLOBAL_RAG_SYNTHESIZER: Union[Any, None] = None 
GLOBAL_CLASSIFIER: Union[Any, None] = None 
GLOBAL_RAG_TOOL_INSTANCE: Union[AdvancedRAGTool, None] = None

# ==============================================================================
# 4. FASTAPI APP SETUP 
# ==============================================================================
app = FastAPI(title="BioStellar Conditional RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    if not CEREBRAS_API_KEY:
        logger.warning("CEREBRAS_API_KEY not set. RAG Chain will not be fully operational.")
    else:
        # 1. Initialize LLM
        GLOBAL_LLM = ChatCerebras(
            model="llama-3.3-70b", 
            api_key=CEREBRAS_API_KEY,
            temperature=0.7,
            max_tokens=2048
        )
        
        # 2. Initialize the Advanced RAG tool instance (used for retrieval)
        GLOBAL_RAG_TOOL_INSTANCE = AdvancedRAGTool(llm=GLOBAL_LLM) 
        
        # 3. Create the Synthesis Chain (Prompt + LLM)
        GLOBAL_RAG_SYNTHESIZER = SYNTHESIS_PROMPT | GLOBAL_LLM

        # 4. Create the Classifier Chain (Classifier Prompt + LLM)
        GLOBAL_CLASSIFIER = CLASSIFIER_PROMPT | GLOBAL_LLM
    
except Exception as e:
    logger.error(f"Failed to initialize LLM or RAG Chain: {e}", exc_info=True)

# ==============================================================================
# 5. CONNECTION MANAGER AND ASSISTANT INVOCATION
# ==============================================================================

class ConnectionManager:
    """Manages active WebSocket connections and per-user chat history."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_chat_history: Dict[str, List[Union[HumanMessage, AIMessage]]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id not in self.user_chat_history:
            self.user_chat_history[user_id] = [] 
        logger.info(f"User {user_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"User {user_id} disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()


async def invoke_assistant(user_message: str, chat_history: List[Union[HumanMessage, AIMessage]]) -> Dict[str, Any]:
    """
    Executes the two-path assistant logic: 
    1. Classification (LLM) 
    2. Synthesis (Conversational) OR Full RAG (Retrieval + Synthesis)
    """
    if not GLOBAL_RAG_SYNTHESIZER or not GLOBAL_RAG_TOOL_INSTANCE or not GLOBAL_CLASSIFIER:
        return {"answer": "System Error: The assistant is not configured correctly (API Key missing).", "mode": "Error"}
    
    try:
        # Step 1: Classification
        classification_result = await GLOBAL_CLASSIFIER.ainvoke({"input": user_message})
        needs_rag = classification_result.content.strip().upper() == 'YES'
        
        context = "No external context required."
        mode = "Conversational"
        num_chunks = 0

        if needs_rag:
            # Step 2A: Retrieval (GLOBAL_RAG_TOOL_INSTANCE now returns a single formatted context string)
            retrieved_context_string = GLOBAL_RAG_TOOL_INSTANCE._run(user_message)
            mode = "RAG"
            
            if retrieved_context_string.startswith("Error:"):
                 context = retrieved_context_string # Propagate error as context
            elif retrieved_context_string.startswith("No relevant context"):
                 context = retrieved_context_string
            else:
                context = retrieved_context_string
                # Count the number of chunks for the status message
                num_chunks = context.count("--- Chunk ")
                

        # Step 2B/3: Synthesis (Using the same synthesizer for both modes)
        result = await GLOBAL_RAG_SYNTHESIZER.ainvoke({
            "input": user_message,
            "context": context,
            "chat_history": chat_history 
        })
        
        return {"answer": result.content, "mode": mode, "chunk_count": num_chunks}
        
    except Exception as e:
        logger.error(f"Error invoking assistant chain: {e}", exc_info=True)
        return {"answer": f"I apologize, but the AI system encountered a critical error: {str(e)}", "mode": "Error", "chunk_count": 0}


# ==============================================================================
# 6. FASTAPI ENDPOINTS
# ==============================================================================

@app.get("/")
async def read_root():
    return {"message": "Cerebras RAG Chain WebSocket backend running. Check /health."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "connections": len(manager.active_connections),
        "llm_ready": GLOBAL_LLM is not None,
        "chain_ready": GLOBAL_RAG_SYNTHESIZER is not None,
        "classifier_ready": GLOBAL_CLASSIFIER is not None
    }

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "").strip()
            
            if not user_message:
                continue
            
            chat_history = manager.user_chat_history.get(user_id, [])
            
            await manager.send_personal_message(
                json.dumps({"type": "typing", "message": "Assistant is classifying the query..."}),
                websocket
            )
            
            # 1. Invoke the Assistant
            assistant_result = await invoke_assistant(user_message, chat_history)
            
            ai_response_text = assistant_result["answer"]
            mode = assistant_result["mode"]
            chunk_count = assistant_result["chunk_count"]
            
            # 2. Send the Status Confirmation
            if mode == "RAG":
                status_text = f"RAG Chain executed: Retrieved {chunk_count} reranked chunks for synthesis."
            elif mode == "Conversational":
                status_text = "Conversational Mode: Answered using general knowledge."
            else:
                 status_text = "Error in processing."


            await manager.send_personal_message(
                json.dumps({
                    "type": "retrieval_status",
                    "message": status_text,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }),
                websocket
            )

            # 3. Update Chat History
            chat_history.append(HumanMessage(content=user_message))
            chat_history.append(AIMessage(content=ai_response_text))
            
            if len(chat_history) > 20: 
                chat_history = chat_history[-20:]
            
            manager.user_chat_history[user_id] = chat_history
            
            # 4. Send final response
            timestamp = datetime.datetime.utcnow().isoformat()
            response_data = {
                "type": "ai_message",
                "message": ai_response_text,
                "timestamp": timestamp,
            }
            
            await manager.send_personal_message(
                json.dumps(response_data),
                websocket
            )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}", exc_info=True)
        await websocket.send_text(json.dumps({"error": "An unexpected server error occurred."}))
        manager.disconnect(websocket, user_id)

if __name__ == "__main__":
    import uvicorn
    logger.info("Server running. Optimized for conditional RAG execution.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
