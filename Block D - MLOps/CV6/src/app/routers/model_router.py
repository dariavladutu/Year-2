"""Router for managing machine learning models."""

from fastapi import APIRouter
from utils.model_utils import list_local_models

router = APIRouter(prefix="/model_management", tags=["Model Management"])


@router.get("/models", summary="List available *.h5 models in /models")
async def get_models() -> list[str]:
    """List all local models available in the /models directory."""
    return list_local_models()
