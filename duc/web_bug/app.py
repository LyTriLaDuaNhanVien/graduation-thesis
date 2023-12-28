from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI(title="Dashboard API", version="0.1.0", description="Dashboard API")
from routers import sql_injection

origins = [
    "http://localhost",
    "http://localhost:9000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")


app.include_router(sql_injection.app, prefix="/sql_injection", tags=["SQL Injection"])
