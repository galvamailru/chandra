"""
Chandra OCR HTTP service.
Exposes OCR (PDF/images) via REST API using chandra-ocr (datalab-to/chandra).
"""
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from chandra.input import load_file
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Chandra OCR API",
    description="OCR service for documents (handwriting, tables, math, forms) — [datalab-to/chandra](https://github.com/datalab-to/chandra)",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

# Lazy-loaded manager (HF method for single-container deploy)
_manager: InferenceManager | None = None


def get_manager() -> InferenceManager:
    global _manager
    if _manager is None:
        _manager = InferenceManager(method="hf")
    return _manager


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".bmp",
}


def _chunk_to_dict(chunk: Any, page_num: int) -> Dict[str, Any]:
    """Convert a chunk (object or dict) to a serializable structure: type, page, bbox, text."""
    bbox: List[float] | None = None
    text = ""
    kind = "unknown"
    if isinstance(chunk, dict):
        bbox = chunk.get("bbox") or chunk.get("box")
        text = chunk.get("text") or chunk.get("content") or chunk.get("markdown") or ""
        kind = chunk.get("type") or chunk.get("category") or kind
    else:
        for attr in ("text", "content", "markdown"):
            if hasattr(chunk, attr):
                val = getattr(chunk, attr, None)
                if val is not None:
                    text = str(val)
                    break
        for attr in ("bbox", "box", "bounding_box"):
            if hasattr(chunk, attr):
                val = getattr(chunk, attr, None)
                if val is not None:
                    bbox = val if isinstance(val, list) else list(val) if hasattr(val, "__iter__") else None
                    break
        for attr in ("type", "category", "kind"):
            if hasattr(chunk, attr):
                val = getattr(chunk, attr, None)
                if val is not None:
                    kind = str(val)
                    break
    if bbox is not None and not isinstance(bbox, list):
        bbox = list(bbox) if hasattr(bbox, "__iter__") else None
    if bbox and len(bbox) >= 4:
        bbox = [round(float(x), 2) for x in bbox[:4]]
    return {"page": page_num, "type": kind, "bbox": bbox, "text": (text or "")[:500]}


def _run_ocr(file_path: Path, page_range: str | None = None) -> Dict[str, Any]:
    """Run Chandra OCR on a file; returns merged markdown, HTML, structure, and metadata."""
    config: Dict[str, Any] = {}
    if page_range:
        config["page_range"] = page_range

    images = load_file(str(file_path), config)
    if not images:
        return {
            "markdown": "",
            "html": "",
            "structure": [],
            "pages": [],
            "num_pages": 0,
            "error": "No pages could be loaded from the file",
        }

    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
    manager = get_manager()
    results = manager.generate(
        batch,
        include_images=True,
        include_headers_footers=False,
    )

    all_markdown: List[str] = []
    all_html: List[str] = []
    pages_meta: List[Dict[str, Any]] = []
    structure: List[Dict[str, Any]] = []
    total_tokens = 0

    for page_num, result in enumerate(results):
        all_markdown.append(result.markdown or "")
        all_html.append(result.html or "")
        total_tokens += result.token_count or 0
        pages_meta.append({
            "page_num": page_num + 1,
            "token_count": result.token_count,
            "num_chunks": len(result.chunks) if result.chunks else 0,
            "num_images": len(result.images) if result.images else 0,
        })
        if result.error:
            pages_meta[-1]["error"] = result.error
        if result.chunks:
            for ch in result.chunks:
                structure.append(_chunk_to_dict(ch, page_num + 1))

    return {
        "markdown": "\n\n".join(all_markdown),
        "html": "\n\n".join(all_html),
        "structure": structure,
        "pages": pages_meta,
        "num_pages": len(results),
        "total_token_count": total_tokens,
    }


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "chandra-ocr"}


@app.get("/")
def index():
    return JSONResponse(
        status_code=307,
        headers={"Location": "/static/index.html"},
        content={},
    )


@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    page_range: str | None = Query(None, description="Page range for PDFs, e.g. 1-5,7,9-12"),
):
    """
    Upload a PDF or image file; returns OCR result as markdown, HTML, and metadata.
    Optional query param: page_range (e.g. "1-5,7,9-12") for PDFs.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            },
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        result = _run_ocr(tmp_path, page_range=page_range)
        result["filename"] = file.filename
        return result
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={"error": f"OCR failed: {exc!s}"},
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
