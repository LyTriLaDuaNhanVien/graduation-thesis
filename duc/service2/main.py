from fastapi import FastAPI
from typing import Dict

import somethingnotexist

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
