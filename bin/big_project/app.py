from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from .chatbot import SchoolChatbot

app = FastAPI(title="School Chatbot API")

# Configure CORS for the web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = SchoolChatbot()

class ChatRequest(BaseModel):
    question: str
    conversation_id: str = None  # Optional, for conversation history

class ChatResponse(BaseModel):
    answer: str
    sources: list = []  # References to where the information came from
    
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        answer, sources = chatbot.answer_question(
            request.question, 
            conversation_id=request.conversation_id
        )
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)