from fastapi import FastAPI

app = FastAPI(
    title="Enterprise RAG Document Intelligence",
    description="A Retrieval-Augmented Generation API for intelligent document question answering.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG Document Intelligence API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
