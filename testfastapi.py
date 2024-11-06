from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI is working!"}

# Run with: uvicorn test_fastapi:app --reload
