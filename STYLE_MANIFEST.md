# Manifesto Visual — SLM from Scratch

Todo relatório, notebook e documento deste repositório segue este guia.

---

## Paleta Primária

| Nome | HEX | RGB | Uso |
|---|---|---|---|
| **Azul Escuro** | `#1D4F91` | `29, 79, 145` | Cor principal — headers, links, bordas de destaque, CTAs |
| **Azul Médio** | `#426DA9` | `66, 109, 169` | Complemento — backgrounds secundários, estado hover |
| **Roxo** | `#77127B` | `119, 18, 123` | Destaque alternativo — badges especiais |
| **Framboesa** | `#C1188B` | `193, 24, 139` | Ênfase — labels de destaque, alertas de atenção |
| **Magenta** | `#E80070` | `232, 0, 112` | Chamada forte — CTA crítico, tag de erro severo |
| **Branco** | `#FFFFFF` | `255, 255, 255` | Fundo de página, texto sobre cores escuras |

---

## Paleta Secundária

Tons derivados das cores primárias — para superfícies, bordas e estados.

| Token | HEX | Uso |
|---|---|---|
| `Blue 10` | `#F7F8FB` | Superfície de card, fundo de célula de notebook |
| `Blue 20` | `#E1E4EF` | Borda sutil, divisor de seção, grid de plot |
| `Blue 30` | `#C6CBE1` | Borda de foco, separador de tabela |
| `Blue 50` | `#8795C0` | Texto de suporte sobre fundo claro |
| `Grey 10` | `#F8F8F9` | Fundo alternativo neutro |
| `Grey 50` | `#8F97A0` | Labels secundários, metadados, timestamps |

---

## Paleta de Status e Dados

Reservada para estado de resultado e visualização de dados — gráficos, barras, heatmaps.

| Nome | HEX | Uso |
|---|---|---|
| `Navy` | `#1A2B4B` | Texto principal, fundo escuro, eixo de gráfico |
| `Sapphire` | `#3087AA` | Série 2 em gráficos |
| `Teal` | `#54C2B8` | Série 3, indicador positivo alternativo |
| `Lime` | `#AFB904` | Série 4, destaque de dado relevante |
| `Alert Green` | `#0FAC67` | Val loss baixo, resultado bom, sucesso |
| `Alert Red` | `#FA1320` | Erro, overfitting, resultado crítico |
| `Orange` | `#FF9735` | Atenção intermediária |
| `Yellow` | `#FEDB00` | Resultado regular, aviso |
| `Grey` | `#495765` | Texto de corpo, labels de eixo |

---

## Tipografia

| Fonte | Contexto |
|---|---|
| **Roboto** | Materiais online — notebooks, relatórios HTML, dashboards |
| **Arial** | Fallback universal — ambientes sem fontes customizadas |
| **Roboto Mono** | Código inline e blocos de código |

```css
font-family: "Roboto", "Arial", sans-serif;
font-family: "Roboto Mono", "Courier New", monospace;
```

---

## Squircles

Todos os elementos de caixa usam o padrão **squircle** com raio calculado por fórmula.

### Fórmula

| Formato | Regra | Exemplo |
|---|---|---|
| **Squircle** (elemento aprox. quadrado) | `lado ÷ 4 = raio` | `100×100px` → `radius: 25px` |
| **Retângulo** (elemento alongado) | `lado menor ÷ 10 = raio` | `200×48px` → `radius: 5px` |

### Tokens de raio padrão

```css
--radius-sm:   8px;    /* badges inline         (32px ÷ 4)  */
--radius-md:   20px;   /* callouts, células     (80px ÷ 4)  */
--radius-lg:   32px;   /* cards principais      (128px ÷ 4) */
--radius-xl:   56px;   /* squircle hero / pill  (224px ÷ 4) */
--radius-rect: 5px;    /* retângulos alongados  (50px ÷ 10) */
```

Nunca usar `border-radius: 4px` ou `8px` em elementos quadrados — quebra o padrão squircle.
Nunca usar `border-radius: 50%` — fica circular, não squircle.

---

## Componentes HTML (relatórios exportados)

Importar `assets/style.css` em todo HTML gerado:

```html
<link rel="stylesheet" href="../assets/style.css">
```

### Card

```html
<div class="card">Conteúdo padrão</div>
<div class="card-primary">Destaque primário (Azul Escuro)</div>
<div class="card-subtle">Informação sutil (Blue 10)</div>
```

### Callouts

```html
<div class="callout callout-info">    Informação neutra   </div>
<div class="callout callout-success"> Resultado positivo  </div>
<div class="callout callout-warning"> Atenção / cuidado   </div>
<div class="callout callout-danger">  Erro / overfitting  </div>
```

### Badges de métrica

```html
<span class="badge badge-blue">      Train loss: 0.18  </span>
<span class="badge badge-green">     Val loss: 1.01    </span>
<span class="badge badge-yellow">    Overfitting        </span>
<span class="badge badge-red">       Loss divergindo    </span>
<span class="badge badge-purple">    Experimento novo   </span>
<span class="badge badge-raspberry"> Atenção especial   </span>
```

### Grid de métricas

```html
<div class="metric-grid">
  <div class="metric-box">
    <div class="value">0.18</div>
    <div class="label">Train Loss</div>
  </div>
  <div class="metric-box">
    <div class="value">936K</div>
    <div class="label">Parâmetros</div>
  </div>
</div>
```

---

## Plots Python (notebooks)

Importar o tema no início de cada notebook:

```python
import sys
sys.path.insert(0, "..")
from assets.theme import apply, COLORS, squircle_ax, loss_plot

apply()  # aplica globalmente
```

### Plot de loss padrão

```python
fig = loss_plot(
    train_losses=[5.3, 3.2, 2.1, ...],
    val_losses=[1.6, 1.1, 1.0, ...],
    eval_interval=500,
    title="Treino — Endereços"
)
fig.savefig("reports/loss_curve.png", dpi=150, bbox_inches="tight")
```

### Cores nos plots

```python
# Sempre referenciar via dicionário, nunca hardcodar hex
ax.plot(x, y, color=COLORS["dark_blue"],   label="Treino")
ax.plot(x, y, color=COLORS["yellow"],      label="Validação", linestyle="--")
ax.axvline(best_step, color=COLORS["alert_green"], label="Melhor checkpoint")
```

---

## Convenções de Markdown

- **Títulos H2** sempre com separador `---` antes da próxima seção
- **Tabelas** sempre com cabeçalho — nunca tabelas sem `thead`
- **Código inline** para: nomes de arquivo, parâmetros, valores numéricos de config, HEX de cores
- **Blocos de código** com linguagem especificada: ` ```python `, ` ```bash `, ` ```yaml `, ` ```css `
- **Nunca** usar emojis decorativos em documentos técnicos

---

## Referências de arquivo

| Arquivo | Papel |
|---|---|
| `assets/style.css` | CSS completo para HTML reports |
| `assets/theme.py` | Tema matplotlib + helpers de plot |
| `STYLE_MANIFEST.md` | Este documento — fonte da verdade visual |
| `CLAUDE.md` | Arquitetura e comandos do projeto |
