# Glossário — SLM from Scratch

Termos técnicos explicados para quem está chegando agora no tema de modelos de linguagem.

---

## O Modelo e sua Arquitetura

**Transformer**
Arquitetura de rede neural criada em 2017 que é a base de todos os modelos de linguagem modernos (GPT, BERT, LLaMA). O princípio central é o mecanismo de atenção: em vez de processar texto palavra por palavra em sequência, o modelo olha para todas as palavras ao mesmo tempo e aprende quais têm relação entre si.

**LLM — Large Language Model**
Modelo de linguagem grande. Um transformer treinado em quantidades massivas de texto (centenas de bilhões de tokens). Exemplos: GPT-4, Claude, LLaMA. Requer infraestrutura de datacenter para treinar.

**SLM — Small Language Model**
Modelo de linguagem pequeno. Mesmo princípio do LLM, mas com arquitetura reduzida (milhões de parâmetros, não bilhões). Treina em GPU doméstica. Este repositório constrói um SLM do zero.

**Decoder-only**
Variante do transformer que só usa a metade "decodificadora" da arquitetura original. É o padrão de todos os modelos generativos (GPT, LLaMA). O modelo vê o texto da esquerda para a direita e prevê o próximo token — nunca "olha para o futuro".

**Parâmetros**
Os números internos do modelo — os pesos de todas as matrizes de transformação. Um modelo de `1M` parâmetros tem 1 milhão de números ajustáveis. Treinar = encontrar os valores ideais para esses números.

**`d_model`**
Dimensão do embedding — o tamanho do vetor que representa cada token internamente. Quanto maior, mais o modelo consegue "pensar" sobre cada token. O principal controle de capacidade do modelo.

**`n_layers`**
Número de blocos transformer empilhados. Cada bloco refina a representação do texto. Mais camadas = mais capacidade de raciocínio, mas mais lento e pesado.

**`n_heads`**
Número de cabeças de atenção. Cada cabeça aprende um tipo diferente de relação entre tokens (sintaxe, semântica, coreference, etc.). Divide `d_model` em `n_heads` sub-espaços paralelos.

**`d_ff`**
Dimensão da camada feed-forward dentro de cada bloco. Geralmente `4 × d_model`. É onde o modelo armazena "conhecimento factual" — os fatos que aprendeu durante o treino.

**Weight tying**
Técnica onde a matriz de embedding de entrada compartilha os mesmos pesos da camada de saída. Reduz parâmetros e melhora generalização sem custo de qualidade.

**LayerNorm / Pre-norm**
Normalização aplicada antes de cada sub-camada (atenção e feed-forward). Estabiliza o treinamento ao manter os valores em escala razoável. "Pre-norm" = normaliza antes, não depois.

---

## Tokenização

**Token**
A unidade básica de texto que o modelo processa. Não é necessariamente uma palavra: pode ser parte de uma palavra (`##ção`), uma palavra inteira (`casa`), um sinal de pontuação (`.`) ou um espaço. O modelo nunca vê letras diretamente — só tokens representados como números inteiros.

**Tokenizador**
Algoritmo que converte texto em uma sequência de tokens (números) e vice-versa. É treinado separadamente do modelo e define o vocabulário que o modelo vai usar.

**BPE — Byte Pair Encoding**
Algoritmo de tokenização. Começa com todos os caracteres individuais e iterativamente funde os pares mais frequentes em tokens novos. Após o treino, palavras comuns viram um único token (`modelo`), palavras raras viram múltiplos (`tran|sfor|mer`). É o algoritmo usado por GPT-2, GPT-4, LLaMA e neste projeto.

**Vocabulário (`vocab_size`)**
Conjunto de todos os tokens que o modelo conhece. Definido pelo tokenizador antes do treino. Este projeto usa `1024` tokens — modelos maiores usam `32.000` a `128.000`.

**`vocab.json`**
Arquivo que mapeia cada string de token para seu ID numérico. Exemplo: `"casa" → 312`.

**`merges.json`**
Arquivo com as regras de fusão do BPE em ordem de prioridade. Durante a codificação de texto novo, as fusões são aplicadas nessa ordem.

**Context length / Janela de contexto**
Quantidade máxima de tokens que o modelo processa de uma vez. Um modelo com `context_length: 128` só "enxerga" os últimos 128 tokens ao gerar o próximo. GPT-4 tem contexto de 128.000 tokens.

---

## Treinamento

**Loss / Função de perda**
Número que mede o erro do modelo. Quanto menor, melhor. Durante o treino, o objetivo é minimizar esse número. Este projeto usa **cross-entropy loss**: o modelo precisa acertar qual token vem depois dado o contexto anterior.

**`train/loss`**
Loss calculado nos dados de treino. Sempre cai ao longo do tempo porque o modelo está sendo otimizado exatamente sobre esses exemplos.

**`val/loss`**
Loss calculado em dados que o modelo nunca viu durante o treino. É o indicador real de qualidade — se cair junto com o `train/loss`, o modelo está aprendendo de verdade.

**Overfitting**
Quando o modelo "decora" os dados de treino em vez de aprender padrões generalizáveis. Sinal: `train/loss` continua caindo mas `val/loss` começa a subir. O checkpoint com menor `val/loss` é sempre o melhor para uso, mesmo não sendo o último.

**Batch / `batch_size`**
Quantidade de exemplos processados em paralelo por passo de treino. Batch maior = gradientes mais estáveis, mas mais memória de GPU.

**Learning rate / Taxa de aprendizado**
Controla o tamanho do passo de cada atualização de pesos. Muito alto: o modelo oscila e não converge. Muito baixo: o treino é lento ou fica preso. Tipicamente na ordem de `0.0003` a `0.001`.

**Warmup**
Período inicial do treino onde o learning rate começa baixo e sobe gradualmente até o valor definido. Evita atualizações bruscas nos primeiros passos quando os pesos ainda são aleatórios.

**Cosine schedule**
Estratégia de aprendizado onde o learning rate sobe durante o warmup e depois decai suavemente seguindo uma curva cosseno até o final do treino. Produz convergência mais suave do que manter a taxa fixa.

**Gradient clipping**
Limita o tamanho máximo do gradiente antes de cada atualização. Evita que um batch ruim cause uma atualização catastrófica que destrua o que o modelo aprendeu até ali.

**AdamW**
Otimizador padrão para transformers. Uma variação do Adam que separa a regularização (`weight_decay`) da adaptação do gradiente. Quase sempre a primeira escolha para treinar LLMs e SLMs.

**Weight decay**
Regularização que penaliza pesos grandes, empurrando-os suavemente para zero. Ajuda a evitar overfitting sem precisar reduzir o modelo.

**Checkpoint**
Arquivo `.pt` com o estado completo do modelo (pesos, estado do otimizador, step atual). Permite pausar e retomar o treino, ou carregar o modelo para inferência sem retreinar.

**`max_steps`**
Número total de passos de treino. Um passo = processar um batch e atualizar os pesos uma vez.

---

## Atenção e Geração

**Self-Attention / Atenção**
Mecanismo central do transformer. Para cada token, calcula quão relevante cada outro token do contexto é para entendê-lo. O resultado é uma média ponderada das representações de todos os tokens, onde os pesos são aprendidos.

**Causal mask / Máscara causal**
Restrição aplicada na atenção para que o modelo só possa "ver" tokens anteriores ao gerar o próximo. Sem essa máscara, o modelo "colaria" no token que deveria prever.

**Flash Attention**
Implementação otimizada do mecanismo de atenção que usa menos memória de GPU e é mais rápida. Disponível no PyTorch >= 2.0 via `F.scaled_dot_product_attention`.

**Autoregressive / Autoregressivo**
Forma de geração onde o modelo produz um token por vez, e cada token gerado entra como entrada para gerar o próximo. É como o modelo "escreve" texto: token por token, da esquerda para a direita.

**Temperature**
Controla a aleatoriedade da geração. `temperature=1.0`: distribuição original. `temperature<1.0` (ex.: `0.3`): mais determinístico, escolhe tokens mais prováveis. `temperature>1.0`: mais criativo, mais aleatório.

**Top-k sampling**
Antes de amostrar o próximo token, descarta todos menos os `k` tokens mais prováveis. `top_k=20` com `temperature=0.3` produz saídas focadas e repetíveis — ideal para normalização de endereços.

---

## Avaliação e Visualização

**Inferência**
Uso do modelo para produzir saídas (gerar texto, normalizar endereços) sem atualizar os pesos. Não treina — só lê.

**Perplexidade**
Métrica derivada da loss: `perplexidade = e^loss`. Mede quão "surpreso" o modelo fica com o texto. Perplexidade `1.0` = prevê tudo perfeitamente. Perplexidade `∞` = não sabe nada.

**TensorBoard**
Ferramenta de visualização de métricas de treino em tempo real. Plota `train/loss`, `val/loss`, `learning_rate` e outras métricas em gráficos interativos no browser.

**wandb — Weights & Biases**
Plataforma de rastreamento de experimentos. Registra hiperparâmetros, métricas e artefatos de cada run. Suporta modo `offline` (salva local, sincroniza depois).

**Netron**
Visualizador de arquitetura de redes neurais. Abre um arquivo `.onnx` e exibe o grafo completo do modelo — camadas, dimensões, conexões — de forma interativa no browser.

**ONNX — Open Neural Network Exchange**
Formato padrão para exportar modelos entre frameworks (PyTorch → ONNX → TensorFlow, etc.). Neste projeto é usado para exportar o modelo e visualizá-lo no Netron.

---

## Referências de arquivo

| Arquivo | Papel |
|---|---|
| `slm/tokenizer.py` | Implementação do BPE |
| `slm/model.py` | Arquitetura do transformer |
| `slm/trainer.py` | Loop de treino e métricas |
| `STYLE_MANIFEST.md` | Padrão visual do projeto |
| `CLAUDE.md` | Arquitetura e comandos |
