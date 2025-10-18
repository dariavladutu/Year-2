"""Plant Root Analysis API.

This FastAPI application provides endpoints for plant root segmentation and analysis.
It includes functionality for image segmentation, root measurement, 
and serving a static HTML UI.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers.segmentation import router as segmentation_router
from app.routers.model_router import router as model_router

app = FastAPI(title="Plant Root Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ── Register API endpoints first ─────────────────────────────────────
app.include_router(segmentation_router)
app.include_router(model_router)

# ── Serve the HTML UI as a fallback for everything else ──────────────
app.mount("/", StaticFiles(directory="frontend_simple", html=True), name="static")
