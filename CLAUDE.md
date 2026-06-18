# CLAUDE.md — SLM from Scratch

Guia de arquitetura, comandos e convenções do repositório. Padrão visual: `STYLE_MANIFEST.md`.

---

## Projeto

Transformer decoder-only (estilo GPT) treinado do zero em PyTorch para normalização de endereços brasileiros. A tarefa é completamento de texto: dado `ENTRADA: {endereço sujo} | SAIDA:`, o modelo completa a forma canônica.

GPU: NVIDIA RTX 2070 (`8 GB` VRAM). `Python 3.12`. Todos os scripts rodam com `PYTHONUTF8=1` no Windows.

---

## Pipeline completo — do zero ao modelo

```
dados brutos (.txt)
      │
      ▼
scripts/make_address_data.py        ← gera pares sintéticos ENTRADA/SAIDA
      │  data/raw/addresses.txt
      ▼
scripts/prepare_data.py             ← treina BPE + converte para .bin
      │  data/tokenizer/vocab.json
      │  data/tokenizer/merges.json
      │  data/processed/train.bin
      │  data/processed/val.bin
      ▼
scripts/train.py --config configs/address.yaml
      │  checkpoints/address/step_XXXXXX.pt
      │  runs/address/  (TensorBoard)
      │  wandb/offline-run-*/  (W&B offline)
      ▼
scripts/address_eval.py             ← inferência interativa
scripts/visualize.py                ← Netron (arquitetura) + W&B sync
```

---

## Comandos

### Preparação de dados

```bash
# Dataset sintético de endereços (gera data/raw/addresses.txt)
python scripts/make_address_data.py

# Treinar tokenizador BPE + converter para binário
python scripts/prepare_data.py \
  --input data/raw/addresses.txt \
  --vocab-size 1024 \
  --output-dir data/processed \
  --tokenizer-dir data/tokenizer
```

Para dados reais: coloque o arquivo `.txt` em `data/raw/`, ajuste `--vocab-size` (recomendado: `2x` o número de tokens únicos estimados) e rode `prepare_data.py`.

### Treino

```bash
# Treinar (configs disponíveis: address.yaml, base.yaml, small.yaml)
python -u scripts/train.py --config configs/address.yaml

# Retomar de checkpoint
python -u scripts/train.py --config configs/address.yaml --resume checkpoints/address/step_010000.pt
```

`-u` desativa o buffer de stdout para ver logs em tempo real.

### Avaliação e geração

```bash
# Modo interativo de normalização de endereços
python scripts/address_eval.py --checkpoint checkpoints/address/step_002000.pt

# Geração genérica com prompt
python scripts/generate.py \
  --checkpoint checkpoints/address/step_002000.pt \
  --prompt "ENTRADA: av paulista 1578 ap 42 sp | SAIDA:" \
  --temperature 0.3 --top-k 20
```

### Visualização

```bash
# TensorBoard (métricas de treino ao vivo)
tensorboard --logdir runs           # → http://localhost:6006

# Netron (arquitetura do modelo em grafo)
python scripts/visualize.py --checkpoint checkpoints/address/step_002000.pt  # → http://localhost:8080

# Só exportar ONNX sem abrir browser
python scripts/visualize.py --checkpoint checkpoints/address/step_002000.pt --no-browser

# Sincronizar runs offline do W&B para a nuvem
python scripts/visualize.py --wandb-sync
```

### API REST (produção em tempo real)

```bash
# Iniciar servidor (porta 8000)
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

# Com checkpoint e tokenizador customizados
CHECKPOINT=checkpoints/address/step_002000.pt \
TOKENIZER_DIR=data/tokenizer \
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

# Endpoints disponíveis:
#   GET  /health                → status + device
#   POST /normalizar            → { endereco } → { original, normalizado }
#   POST /normalizar/batch      → { enderecos: [...] } → { resultados: [...] }
```

### PySpark (processamento em lote / big data)

```bash
# Standalone — lê CSV, escreve Parquet
spark-submit scripts/spark_udf.py \
  --input  data/enderecos.csv \
  --output data/enderecos_norm \
  --col    endereco_bruto \
  --partitions 8

# Como UDF em pipeline existente
python - <<'EOF'
from scripts.spark_udf import make_normalizar_udf
from pyspark.sql.functions import col

normalizar = make_normalizar_udf(
    checkpoint    = "checkpoints/address/step_002000.pt",
    tokenizer_dir = "data/tokenizer",
)
df = df.withColumn("endereco_norm", normalizar(col("endereco_bruto")))
EOF
```

---

## Arquitetura do modelo (`slm/model.py`)

Transformer decoder-only com as seguintes escolhas de design:

- **Pre-norm**: LayerNorm aplicado antes de cada sub-camada (mais estável que post-norm)
- **Weight tying**: `tok_emb.weight == head.weight` — reduz parâmetros e melhora generalização
- **Flash Attention**: usa `F.scaled_dot_product_attention` quando disponível (PyTorch >= 2.0), com fallback manual
- **Causal mask**: registrada como buffer (não é parâmetro), tamanho fixo em `context_length`

Hierarquia: `SLM` → `Block` × `n_layers` → `CausalSelfAttention` + `FeedForward`

`SLM.forward(idx, targets=None)` retorna `(logits, loss)`. Quando `targets=None`, `loss=None` (modo inferência). `SLM.generate()` faz autoregressive sampling com temperatura e top-k opcionais.

---

## Camada de inferência (`slm/inference.py`)

**Fonte única** de carregamento e geração. Os entrypoints (`serve.py`, `spark_udf.py`, `address_eval.py`, `generate.py`) são cascas finas sobre ela — nenhum duplica a lógica de tensorizar/gerar/decodificar.

- `load_model(checkpoint, device)` / `resolve_device(device)` — carregamento e resolução de device (`"auto"` → cuda/cpu)
- `TextGenerator` — geração de texto livre; método `.generate(prompt, ...)`
- `AddressNormalizer(TextGenerator)` — normalização `ENTRADA/SAIDA`; método `.normalize(endereco, ...)` retorna só a forma canônica

```python
from slm.inference import AddressNormalizer
norm = AddressNormalizer.from_checkpoint("checkpoints/address/step_002000.pt", "data/tokenizer")
norm.normalize("av paulista 1578 sp")
```

**Onde integrar DNE / tabelas de lookup**: o pré-processamento (CEP → logradouro, expansão de abreviações) e o pós-processamento (validação de UF/CEP) entram em `AddressNormalizer.normalize` — assim API, Spark e CLI herdam o mesmo fluxo automaticamente.

---

## Tokenizador BPE (`slm/tokenizer.py`)

Implementação do zero, sem dependências externas além de `regex`.

**Dois arquivos persistidos:**
- `vocab.json` — mapeamento `token_string → id_inteiro`. Base: 4 tokens especiais + 256 chars ASCII. Os merges aprendidos são adicionados ao final.
- `merges.json` — lista ordenada de regras `[[token_a, token_b], token_fusão]`. A ordem é a prioridade de aplicação durante encoding.

**Encoding**: para cada palavra (pré-tokenizada pelo padrão GPT-4), aplica merges iterativamente em ordem de `_merge_rank` até não restar candidatos. Resultado: lista de IDs via `vocab`.

**Atenção**: `learning_rate` nos YAMLs deve ser escrito como decimal (`0.0005`), não notação científica (`5e-4`) — o PyYAML não converte para float automaticamente.

---

## Configuração (`configs/`)

Cada YAML mapeia exatamente para `ModelConfig` + `TrainConfig` em `slm/config.py`. Campos relevantes:

| Campo | Onde afeta |
|---|---|
| `vocab_size` | Tamanho do embedding e da camada de saída. Deve ser igual ao tokenizador usado. |
| `context_length` | Tamanho máximo de sequência. Endereços cabem em `128`. |
| `d_model` | Dimensão de embedding — principal alavanca de capacidade. |
| `wandb_mode` | `"offline"` salva localmente; `"online"` requer conta wandb.ai; `"disabled"` pula. |
| `checkpoint_dir` | Nome da subpasta vira o nome do run no TensorBoard e W&B. |

**Configs disponíveis:**
- `address.yaml` — ~`1M` params, context `128`, otimizado para endereços
- `small.yaml` — ~`500K` params, para experimentos rápidos
- `base.yaml` — ~`10M` params, para corpus maiores

---

## Dados reais de endereços

O modelo é treinado como tarefa de completamento. O formato do arquivo de entrada deve ser:

```
ENTRADA: {endereço como recebido} | SAIDA: {forma canônica desejada}
ENTRADA: r das flores 123 sp | SAIDA: Rua das Flores, 123 - São Paulo/SP
ENTRADA: Av. Paulista nº1578 ap 42 Sao Paulo | SAIDA: Avenida Paulista, 1578, Apartamento 42 - São Paulo/SP
```

Uma linha por exemplo, encoding `UTF-8`. Para produção:
1. Substituir `data/raw/addresses.txt` pelo arquivo real
2. Ajustar `--vocab-size` em `prepare_data.py` (`1024`–`4096` dependendo da variedade léxica)
3. Aumentar `max_steps` e reduzir `learning_rate` para datasets maiores
4. Monitorar `val/loss` — parar quando começar a subir (overfitting)

O melhor checkpoint geralmente não é o último: verificar qual step teve menor `val/loss` no TensorBoard e usar esse para inferência.

---

## Checkpoints

Cada `.pt` contém `model`, `optimizer`, `model_cfg` e `train_cfg` — o suficiente para retomar treino ou carregar para inferência sem precisar do YAML original.

```python
ckpt = torch.load("checkpoints/address/step_002000.pt", weights_only=False)
model = SLM(ckpt["model_cfg"])
model.load_state_dict(ckpt["model"])
```

Checkpoints e dados processados não são versionados pelo git (`.gitignore`).

---

## Overfitting com datasets pequenos

Com dados sintéticos (`5k` exemplos), a `val/loss` começa a subir por volta do step `1500`–`2000` enquanto a `train/loss` continua caindo. Isso é esperado. Para dados reais maiores, o comportamento será diferente. O checkpoint com menor `val/loss` é sempre a melhor escolha para produção, independente de ser o último.

---

## Referências de arquivo

| Arquivo | Papel |
|---|---|
| `slm/model.py` | Arquitetura do transformer (SLM, Block, Attention, FF) |
| `slm/tokenizer.py` | BPE tokenizer do zero |
| `slm/trainer.py` | Loop de treino, TensorBoard, W&B |
| `slm/dataset.py` | Dataset com `np.memmap` |
| `slm/config.py` | Dataclasses `ModelConfig` e `TrainConfig` |
| `slm/inference.py` | Camada de inferência única (`AddressNormalizer`, `TextGenerator`) |
| `configs/address.yaml` | Config do modelo de endereços |
| `assets/style.css` | CSS para relatórios HTML exportados |
| `assets/theme.py` | Tema matplotlib + helpers de plot |
| `STYLE_MANIFEST.md` | Padrão visual — fonte da verdade |
| `GLOSSARIO.md` | Termos técnicos explicados para não-especialistas |
