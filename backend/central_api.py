import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from backend.chatbot import CustomRAG  # Make sure to have your rag.py file in the same directory or adjust the import path accordingly.

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI()
load_dotenv()

# Allow CORS for Streamlit frontend
origins = ["http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model
class ChatQuery(BaseModel):
    question: str

# Initialize the CustomRAG instance
custom_rag = CustomRAG()

@app.post("/ask-chatbot")
async def ask_chatbot(query: ChatQuery):
    try:
        response = custom_rag.invoke(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))