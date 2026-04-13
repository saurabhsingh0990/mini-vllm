from fastapi import FastAPI
from pydantic import BaseModel
from src.model.model_loader import ModelLoader

app = FastAPI()
model = ModelLoader()

class Request(BaseModel):
    prompt: str
    max_length: int = 100

@app.post("/generate")
def generate_text(req: Request):
    output = model.generate(req.prompt, req.max_length)
    return {"response": output}