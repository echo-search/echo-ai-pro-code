from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, generate

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = load_model()

@app.post("/generate")
async def generate_code(req: Request):
    data = await req.json()
    prompt = data.get("prompt","")
    code = generate(model, prompt)
    return {"code": code}
