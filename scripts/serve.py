"""
API REST para normalização de endereços.

Uso:
    uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

Variáveis de ambiente:
    CHECKPOINT      caminho do .pt  (padrão: checkpoints/address/step_002000.pt)
    TOKENIZER_DIR   diretório do tokenizador (padrão: data/tokenizer)
    DEVICE          cpu | cuda  (padrão: auto-detecta)
"""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from slm.model import SLM
from slm.tokenizer import BPETokenizer

# ── Estado global (carregado uma vez na inicialização) ───────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint  = os.environ.get("CHECKPOINT",    "checkpoints/address/step_002000.pt")
    tok_dir     = os.environ.get("TOKENIZER_DIR", "data/tokenizer")
    device_arg  = os.environ.get("DEVICE",        "auto")

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_arg == "auto" else device_arg
    )

    tokenizer = BPETokenizer()
    tokenizer.load(tok_dir)

    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SLM(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _state["model"]     = model
    _state["tokenizer"] = tokenizer
    _state["device"]    = device

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


# ── Inferência ───────────────────────────────────────────────────────────
def _run(endereco: str, temperature: float, top_k: int, max_new_tokens: int) -> str:
    tok    = _state["tokenizer"]
    model  = _state["model"]
    device = _state["device"]

    prompt = f"ENTRADA: {endereco.strip()} | SAIDA:"
    ids    = [tok.bos_id] + tok.encode(prompt)
    idx    = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)

    decoded = tok.decode(out[0].tolist())
    if "SAIDA:" in decoded:
        return decoded.split("SAIDA:")[-1].strip().split("\n")[0].strip()
    return decoded.strip()


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": _state.get("device")}


@app.post("/normalizar", response_model=NormalizarResponse)
def normalizar(req: NormalizarRequest):
    if not req.endereco.strip():
        raise HTTPException(status_code=422, detail="endereco não pode ser vazio")
    normalizado = _run(req.endereco, req.temperature, req.top_k, req.max_new_tokens)
    return NormalizarResponse(original=req.endereco, normalizado=normalizado)


@app.post("/normalizar/batch", response_model=BatchResponse)
def normalizar_batch(req: BatchRequest):
    resultados = [
        NormalizarResponse(
            original=e,
            normalizado=_run(e, req.temperature, req.top_k, max_new_tokens=60)
        )
        for e in req.enderecos if e.strip()
    ]
    return BatchResponse(resultados=resultados)
