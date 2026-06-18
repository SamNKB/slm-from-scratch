"""
PySpark UDF para normalização de endereços em lote.

O modelo é carregado uma vez por executor (singleton) e reutilizado
em todos os batches da mesma partição — sem overhead de re-inicialização.

Uso como UDF em pipeline existente:
    from scripts.spark_udf import make_normalizar_udf
    from pyspark.sql.functions import col

    normalizar = make_normalizar_udf(
        checkpoint  = "checkpoints/address/step_002000.pt",
        tokenizer_dir = "data/tokenizer",
    )
    df = df.withColumn("endereco_norm", normalizar(col("endereco_bruto")))

Uso standalone (spark-submit):
    spark-submit scripts/spark_udf.py \\
        --input  data/enderecos.csv  \\
        --output data/enderecos_norm \\
        --col    endereco_bruto
"""

from __future__ import annotations

import os
import sys
import torch
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType

# ── Singleton por executor ───────────────────────────────────────────────
# Chaveado pelo caminho do checkpoint — suporta múltiplos modelos no mesmo job.
_cache: dict = {}


def _load(checkpoint: str, tokenizer_dir: str):
    if checkpoint in _cache:
        return _cache[checkpoint]

    # sys.path precisa incluir a raiz do projeto em cada worker
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from slm.model import SLM
    from slm.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load(tokenizer_dir)

    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SLM(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    _cache[checkpoint] = (model, tokenizer)
    return _cache[checkpoint]


# ── Factory do UDF ───────────────────────────────────────────────────────
def make_normalizar_udf(
    checkpoint:    str   = "checkpoints/address/step_002000.pt",
    tokenizer_dir: str   = "data/tokenizer",
    temperature:   float = 0.3,
    top_k:         int   = 20,
    max_new_tokens: int  = 60,
):
    """
    Retorna um pandas_udf pronto para uso em DataFrames Spark.
    Os parâmetros são capturados no closure — sem objetos não-serializáveis.
    """
    _ckpt    = checkpoint
    _tok_dir = tokenizer_dir
    _temp    = temperature
    _topk    = top_k
    _max     = max_new_tokens

    @pandas_udf(StringType())
    def _udf(series: pd.Series) -> pd.Series:
        model, tok = _load(_ckpt, _tok_dir)
        results = []

        for endereco in series:
            if not endereco or not str(endereco).strip():
                results.append(None)
                continue

            prompt = f"ENTRADA: {str(endereco).strip()} | SAIDA:"
            ids    = [tok.bos_id] + tok.encode(prompt)
            idx    = torch.tensor([ids], dtype=torch.long)

            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=_max,
                                     temperature=_temp, top_k=_topk)

            decoded = tok.decode(out[0].tolist())
            if "SAIDA:" in decoded:
                result = decoded.split("SAIDA:")[-1].strip().split("\n")[0].strip()
            else:
                result = decoded.strip()

            results.append(result)

        return pd.Series(results)

    return _udf


# ── Execução standalone ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalização de endereços com PySpark")
    parser.add_argument("--input",       required=True,  help="CSV ou Parquet de entrada")
    parser.add_argument("--output",      required=True,  help="Parquet de saída")
    parser.add_argument("--col",         default="endereco", help="Coluna de endereço bruto")
    parser.add_argument("--checkpoint",  default="checkpoints/address/step_002000.pt")
    parser.add_argument("--tokenizer",   default="data/tokenizer")
    parser.add_argument("--format",      default="csv", choices=["csv", "parquet"])
    parser.add_argument("--partitions",  type=int, default=8,
                        help="Número de partições (1 executor = 1 modelo carregado)")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("slm-address-normalization")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    if args.format == "csv":
        df = spark.read.csv(args.input, header=True, inferSchema=True)
    else:
        df = spark.read.parquet(args.input)

    # Reparticiona para controlar quantos executores carregam o modelo
    df = df.repartition(args.partitions)

    normalizar = make_normalizar_udf(
        checkpoint    = args.checkpoint,
        tokenizer_dir = args.tokenizer,
    )

    df_out = df.withColumn(f"{args.col}_normalizado", normalizar(col(args.col)))
    df_out.write.mode("overwrite").parquet(args.output)

    total = df_out.count()
    print(f"Processados: {total} registros -> {args.output}")
    spark.stop()
