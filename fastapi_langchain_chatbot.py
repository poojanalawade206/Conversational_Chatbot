import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv

# Import LangChain components
from langchain_openai import ChatOpenAI
# --- IMPORT FIX: These modules are now in 'langchain_classic' ---
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
# ---
from langchain_core.prompts import ChatPromptTemplate

# --- Load Environment Variables ---
# Load .env file (which should contain OPENAI_API_KEY)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file. Please create a .env file and add your key.")

# --- Pydantic Models ---

# Request/Response models for /chat
class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    response: str

# Request/Response models for /context
class ContextRequest(BaseModel):
    context: str

class ContextResponse(BaseModel):
    context: str

class ContextUpdateResponse(BaseModel):
    message: str

# Request/Response models for /sentiment
class SentimentRequest(BaseModel):
    input: str

# This is the internal model for structured output
class SentimentModel(BaseModel):
    """Pydantic model for sentiment analysis."""
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the text")

# This is the public response model
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]

# Request/Response models for /entities
class EntitiesRequest(BaseModel):
    input: str

# Internal model for structured output
class EntitiesModel(BaseModel):
    """Pydantic model for entity extraction."""
    entities: List[str] = Field(description="A list of extracted named entities")

# Public response model
class EntitiesResponse(BaseModel):
    entities: List[str]


# --- FastAPI Application ---

app = FastAPI(
    title="Conversational Chatbot API",
    description="A FastAPI server for a conversational AI chatbot using LangChain and OpenAI.",
    version="1.0.0"
)

# --- App Lifecycle: Model Loading ---

@app.on_event("startup")
async def startup_event():
    """
    On application startup, load all models and chains into the app.state
    for efficient access across requests.
    """
    # Initialize the core LLM
    # We use a modern, cost-effective model
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY, temperature=0.7)
    
    # Store LLM in app state
    app.state.llm = llm

    # 1. Setup for /chat endpoint
    # Initialize conversation memory and chain
    app.state.memory = ConversationBufferMemory()
    app.state.conversation = ConversationChain(
        llm=llm,
        memory=app.state.memory,
        verbose=True  # Set to True for debugging to see chain logs
    )

    # 2. Setup for /sentiment endpoint
    # Create a chain for sentiment analysis using structured output
    sentiment_prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following text: {input}"
    )
    # Bind the LLM to the structured output model
    llm_with_sentiment_tool = llm.with_structured_output(SentimentModel)
    app.state.sentiment_chain = sentiment_prompt | llm_with_sentiment_tool

    # 3. Setup for /entities endpoint
    # Create a chain for entity extraction using structured output
    entities_prompt = ChatPromptTemplate.from_template(
        "Extract all key named entities (people, places, organizations, dates, etc.) from this text: {input}"
    )
    # Bind the LLM to the structured output model
    llm_with_entities_tool = llm.with_structured_output(EntitiesModel)
    app.state.entity_chain = entities_prompt | llm_with_entities_tool


# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    """
    Handle user input and return a response from the chatbot.
    This endpoint uses the persistent conversation chain.
    """
    conversation = request.app.state.conversation
    if not conversation:
        raise HTTPException(status_code=500, detail="Conversation chain not initialized")

    try:
        # Use .ainvoke for asynchronous execution
        result = await conversation.ainvoke(request_body.input)
        return ChatResponse(response=result["response"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

@app.get("/context", response_model=ContextResponse)
async def get_context(request: Request):
    """
    Retrieve the current conversation context (history).
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(status_code=500, detail="Conversation memory not initialized")

    try:
        # Load memory variables asynchronously
        memory_vars = await memory.aload_memory_variables({})
        return ContextResponse(context=memory_vars.get("history", "No context available."))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {e}")

@app.post("/context", response_model=ContextUpdateResponse)
async def set_context(request_body: ContextRequest, request: Request):
    """
    Set (or overwrite) the conversation context.
    This clears the existing memory and injects the new context.
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(status_code=500, detail="Conversation memory not initialized")

    try:
        memory.clear()
        # "Prime" the memory with the new context as if it were a past conversation
        await memory.asave_context(
            {"input": "This is the provided context"}, 
            {"output": request_body.context}
        )
        return ContextUpdateResponse(message="Context updated successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting context: {e}")

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request_body: SentimentRequest, request: Request):
    """
    Analyze the sentiment of a given user input.
    Uses a dedicated, structured-output chain.
    """
    sentiment_chain = request.app.state.sentiment_chain
    if not sentiment_chain:
        raise HTTPException(status_code=500, detail="Sentiment chain not initialized")

    try:
        # The chain will return a Pydantic object (SentimentModel)
        result = await sentiment_chain.ainvoke({"input": request_body.input})
        return SentimentResponse(sentiment=result.sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {e}")

@app.post("/entities", response_model=EntitiesResponse)
async def extract_entities(request_body: EntitiesRequest, request: Request):
    """
    Extract named entities from a given user input.
    Uses a dedicated, structured-output chain.
    """
    entity_chain = request.app.state.entity_chain
    if not entity_chain:
        raise HTTPException(status_code=500, detail="Entity chain not initialized")

    try:
        # The chain will return a Pydantic object (EntitiesModel)
        result = await entity_chain.ainvoke({"input": request_body.input})
        return EntitiesResponse(entities=result.entities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {e}")

# --- Main execution ---

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Access the API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
   