from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .digitalAssistant import DigitalAssistant
from .rag.index import create_vectordb_index

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class UserInput(BaseModel):
    input: str
    sessionId: str

class Index(BaseModel):
    name: str

bot = DigitalAssistant()

@app.post("/chat")
async def get_bot_response(user_input: UserInput):
    # Here you can implement your chatbot logic
    input_text = user_input.input
    session_id = user_input.sessionId
    state = {"question": input_text, "session_id": session_id}
    response = bot.graph.invoke(state)
    # response = f"You said: {input_text}. This is a response from session {session_id}."
    return {"message": response["model_response"]}

@app.post("/create-index")
async def create_index(index_details: Index):
    index_name = index_details.name
    try:
        create_vectordb_index(index_name)
        return {"message": f"Index {index_name} created successfully!"}
    except Exception as e:
        print("This is the error faced while creating an index: ", str(e))
        return {"message": f"An error occurred: {str(e)}"}

@app.post("/add-data")
async def add_data():
    # Add the code to add data to the index

    return {"message": "Data indexed successfully!"}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

