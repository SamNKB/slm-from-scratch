# SLM from Scratch

Transformer decoder-only treinado do zero em PyTorch para normalização de endereços brasileiros.
O modelo aprende a completar `ENTRADA: {endereço bruto} | SAIDA:` com a forma canônica — sem regras,
só estatística aprendida dos dados.

---

## O que está aqui

| Componente | Descrição |
|---|---|
| Arquitetura | Transformer decoder-only (estilo GPT), pré-norm, Flash Attention, weight tying |
| Tokenizador | BPE implementado do zero, sem dependências externas além de `regex` |
| Treinamento | AdamW + cosine LR schedule + warmup, TensorBoard, W&B offline |
| API REST | FastAPI — endpoints `/normalizar` e `/normalizar/batch` |
| Big data | PySpark `pandas_udf` com singleton por executor |
| Visualização | Netron (grafo da rede via ONNX) + TensorBoard + W&B |

---

## Requisitos

- Python `3.12`
- NVIDIA GPU com `8 GB` VRAM (treino) — inferência roda em CPU
- CUDA `12.x`

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Pipeline

```
dados brutos (.txt)
      │
      ▼  scripts/make_address_data.py   ← gera exemplos sintéticos
      │
      ▼  scripts/prepare_data.py        ← treina BPE + converte para .bin
      │
      ▼  scripts/train.py               ← treina o transformer
      │
      ├─ scripts/address_eval.py        ← inferência interativa
      ├─ scripts/serve.py               ← API REST (FastAPI)
      └─ scripts/spark_udf.py           ← normalização em lote (PySpark)
```

---

## Início rápido

```bash
# 1. Gerar dados sintéticos
python scripts/make_address_data.py

# 2. Preparar tokenizador e dataset binário
python scripts/prepare_data.py \
  --input data/raw/addresses.txt \
  --vocab-size 1024 \
  --output-dir data/processed \
  --tokenizer-dir data/tokenizer

# 3. Treinar
python -u scripts/train.py --config configs/address.yaml

# 4. Testar interativamente
python scripts/address_eval.py --checkpoint checkpoints/address/step_002000.pt
```

---

## Deploy

### API REST

```bash
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
```

```bash
# Normalizar um endereço
curl -X POST http://localhost:8000/normalizar \
  -H "Content-Type: application/json" \
  -d '{"endereco": "av paulista 1578 ap 42 sp"}'

# Resposta
# {"original": "av paulista 1578 ap 42 sp", "normalizado": "Avenida Paulista, 1578, Apartamento 42 - São Paulo/SP"}
```

### PySpark (big data)

```python
from scripts.spark_udf import make_normalizar_udf
from pyspark.sql.functions import col

normalizar = make_normalizar_udf(
    checkpoint    = "checkpoints/address/step_002000.pt",
    tokenizer_dir = "data/tokenizer",
)
df = df.withColumn("endereco_normalizado", normalizar(col("endereco_bruto")))
```

---

## Monitoramento

```bash
# TensorBoard
tensorboard --logdir runs                    # → http://localhost:6006

# Netron (grafo da arquitetura)
python scripts/visualize.py --checkpoint checkpoints/address/step_002000.pt  # → http://localhost:8080
```

---

## Estrutura

```
slm/                  núcleo do modelo
  model.py            arquitetura (SLM, Block, Attention, FeedForward)
  tokenizer.py        BPE do zero
  trainer.py          loop de treino
  dataset.py          dataset com np.memmap
  config.py           ModelConfig e TrainConfig

scripts/              executáveis
  train.py            treino
  generate.py         geração com prompt livre
  address_eval.py     avaliação interativa de endereços
  serve.py            API REST (FastAPI)
  spark_udf.py        UDF para PySpark
  visualize.py        Netron + wandb sync
  make_address_data.py gerador de dados sintéticos
  prepare_data.py     tokenizador + conversão para binário

configs/              hiperparâmetros
  address.yaml        ~1M params, context 128
  small.yaml          ~500K params
  base.yaml           ~10M params

assets/               identidade visual
  style.css           CSS para relatórios HTML
  theme.py            tema matplotlib
```

---

## Documentação

| Arquivo | Conteúdo |
|---|---|
| `CLAUDE.md` | Arquitetura, comandos e convenções do projeto |
| `GLOSSARIO.md` | 35 termos técnicos explicados para não-especialistas |
| `STYLE_MANIFEST.md` | Padrão visual — paleta, tipografia, squircles |
