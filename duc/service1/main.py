from fastapi import FastAPI
from typing import Dict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
