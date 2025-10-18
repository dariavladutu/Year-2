"""Segment images into binary masks using a specified model."""
from __future__ import annotations
import io
import zipfile
from flask import json 
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from utils.io_utils import encode_mask_to_tiff_base64
from utils.model_utils import get_model_path_local, load_model
from utils.postprocessing_utils import morphological_closing, threshold_mask
from utils.segmentation_utils import segment_image

router = APIRouter(prefix="/segment", tags=["Segmentation"])

# Figure out repo root (…/CV6) no matter where the file lives
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "CV6" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

TEMP_ROOT = PROJECT_ROOT / "temp_feedback"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@router.post("/", summary="Segment image(s) into binary mask(s)")
async def segment_endpoint(
    files: List[UploadFile] = File(...),
    model_id: str = Form(...),
) -> None:
    """Segment uploaded image(s) using a specified model."""
    if not files:
        raise HTTPException(400, "No files uploaded")

    # ── Load model ─────────────────────────────────────────────────
    try:
        model = load_model(get_model_path_local(model_id))
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(500, f"Model load error: {e}") from e

    # ── Per-request scratch dir ────────────────────────────────────
    sid = uuid.uuid4().hex
    session_dir = TEMP_ROOT / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(json.dumps({"model_id": model_id}))

    out_files: list[tuple[str, bytes]] = []

    for f in files:
        if f.content_type not in {"image/png", "image/jpeg", "image/tiff", "image/tif"}:
            raise HTTPException(415, f"Unsupported type: {f.filename}")

        img = await f.read()
        stem = Path(f.filename).stem
        (session_dir / f"{stem}.png").write_bytes(img)  # optional: save original

        seg = segment_image(model, img)
        closed = morphological_closing(threshold_mask(seg["mask"], 0.1))

        meta = {"crop_info": seg["crop_info"], "pad_info": seg["pad_info"]}
        tiff_bytes, _, _ = encode_mask_to_tiff_base64(closed, meta)

        name = f"{stem}_mask.tif"
        (session_dir / name).write_bytes(tiff_bytes)
        out_files.append((name, tiff_bytes))

    # ── Build response ────────────────────────────────────────────
    headers = {"X-Session-ID": sid}

    if len(out_files) == 1:
        fname, data = out_files[0]
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/octet-stream",
            headers={
                **headers,
                "Content-Disposition": f'attachment; filename="{fname}"',
            },
        )

    # multiple → zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in out_files:
            zf.writestr(name, data)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={**headers, "Content-Disposition": 'attachment; filename="masks.zip"'},
    )
