from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.resume.routes import router as resume_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:5173", 
                   "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume_router, prefix="/resume", tags=["resume"])

@app.on_event("startup")
def on_startup():
    print("🚀 Server is running... Visit http://localhost:8000/docs")
