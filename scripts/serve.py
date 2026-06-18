"""
API REST para normalização de endereços.

Uso:
    uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

Variáveis de ambiente:
    CHECKPOINT      caminho do .pt  (padrão: checkpoints/address/step_002000.pt)
    TOKENIZER_DIR   diretório do tokenizador (padrão: data/tokenizer)
    DEVICE          cpu | cuda | auto  (padrão: auto-detecta)
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from slm.inference import AddressNormalizer

# ── Estado global (carregado uma vez na inicialização) ───────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get("CHECKPOINT",    "checkpoints/address/step_002000.pt")
    tok_dir    = os.environ.get("TOKENIZER_DIR", "data/tokenizer")
    device     = os.environ.get("DEVICE",        "auto")

    _state["normalizer"] = AddressNormalizer.from_checkpoint(checkpoint, tok_dir, device)
    yield


app = FastAPI(title="SLM Address Normalizer", version="1.0.0", lifespan=lifespan)


# ── Schemas ──────────────────────────────────────────────────────────────
class NormalizarRequest(BaseModel):
    endereco:       str
    temperature:    float = 0.3
    top_k:          int   = 20
    max_new_tokens: int   = 60


class NormalizarResponse(BaseModel):
    original:    str
    normalizado: str


class BatchRequest(BaseModel):
    enderecos:   list[str]
    temperature: float = 0.3
    top_k:       int   = 20


class BatchResponse(BaseModel):
    resultados: list[NormalizarResponse]


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(_state["normalizer"].device)}


@app.post("/normalizar", response_model=NormalizarResponse)
def normalizar(req: NormalizarRequest):
    if not req.endereco.strip():
        raise HTTPException(status_code=422, detail="endereco não pode ser vazio")
    normalizado = _state["normalizer"].normalize(
        req.endereco,
        temperature=req.temperature,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
    )
    return NormalizarResponse(original=req.endereco, normalizado=normalizado)


@app.post("/normalizar/batch", response_model=BatchResponse)
def normalizar_batch(req: BatchRequest):
    norm = _state["normalizer"]
    resultados = [
        NormalizarResponse(
            original=e,
            normalizado=norm.normalize(e, temperature=req.temperature, top_k=req.top_k),
        )
        for e in req.enderecos if e.strip()
    ]
    return BatchResponse(resultados=resultados)
