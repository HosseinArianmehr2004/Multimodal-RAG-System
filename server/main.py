import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routes.search_routes import router as search_router

app = FastAPI(title="Multimodal RAG Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
app.mount("/client", StaticFiles(directory="client"), name="client")
app.mount("/content", StaticFiles(directory="content"), name="content")

# Routes
app.include_router(search_router)


@app.get("/", response_class=HTMLResponse)
async def home():
    # Serve the index.html file properly
    html_path = os.path.join("client", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    else:
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
