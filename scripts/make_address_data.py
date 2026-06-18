"""
Gera dataset sintético de normalização de endereços brasileiros.

Saída: data/raw/addresses.txt
Formato por linha:
    ENTRADA: {endereço sujo} | SAIDA: {endereço canônico}

O modelo aprende: dado o prefixo "ENTRADA: ... | SAIDA:", completar o restante.
"""

import random
from pathlib import Path

random.seed(42)

TIPOS = {
    "Rua":      ["Rua", "R.", "r.", "rua", "RUA"],
    "Avenida":  ["Avenida", "Av.", "av.", "avenida", "Ave.", "AVE"],
    "Travessa": ["Travessa", "Tv.", "tv.", "travessa"],
    "Alameda":  ["Alameda", "Al.", "alameda"],
    "Praça":    ["Praça", "Pç.", "Praca", "praca", "Pca."],
    "Rodovia":  ["Rodovia", "Rod.", "rod.", "rodovia"],
}

NOMES = [
    "das Flores", "Paulista", "dos Bandeirantes", "XV de Novembro",
    "da Liberdade", "Sete de Setembro", "das Acácias", "do Comércio",
    "Presidente Vargas", "Jorge Amado", "Marechal Deodoro",
    "Castro Alves", "da República", "das Palmeiras", "Santos Dumont",
    "Tiradentes", "João Pessoa", "do Rosário", "São Francisco",
    "Voluntários da Pátria", "Visconde de Pirajá", "Nossa Senhora",
]

CIDADES = {
    "São Paulo/SP":      ["São Paulo", "Sao Paulo", "S.Paulo", "SP", "S. Paulo"],
    "Rio de Janeiro/RJ": ["Rio de Janeiro", "Rio", "RJ", "rio de janeiro"],
    "Belo Horizonte/MG": ["Belo Horizonte", "BH", "bh", "Belo Hte"],
    "Curitiba/PR":       ["Curitiba", "Ctba", "cwb", "curitiba"],
    "Porto Alegre/RS":   ["Porto Alegre", "POA", "poa", "porto alegre"],
    "Salvador/BA":       ["Salvador", "SSA", "ssa", "salvador/ba"],
    "Fortaleza/CE":      ["Fortaleza", "fort.", "fortaleza", "CE"],
    "Recife/PE":         ["Recife", "REC", "recife", "Recife/PE"],
    "Manaus/AM":         ["Manaus", "MAO", "manaus"],
    "Goiânia/GO":        ["Goiânia", "Goiania", "goiania", "GYN"],
}

COMPLEMENTOS = {
    None:         [("", "")],
    "Apartamento":[(", Apartamento {n}", " apto {n}"), (", Apartamento {n}", " ap{n}"),
                   (", Apartamento {n}", " Apto. {n}"), (", Apartamento {n}", " apartamento {n}")],
    "Sala":       [(", Sala {n}", " sala {n}"), (", Sala {n}", " sl {n}")],
    "Casa":       [(", Casa {n}", " casa {n}"), (", Casa {n}", " cs {n}")],
    "Loja":       [(", Loja {n}", " loja {n}"), (", Loja {n}", " lj{n}")],
}


def _numero_sujo(n: int) -> str:
    return random.choice([
        str(n), f"nº{n}", f"nº {n}", f"n.{n}", f"n° {n}", f"numero {n}", f"Nº {n}", f"N.{n}",
    ])


def gerar_par() -> tuple[str, str]:
    tipo = random.choice(list(TIPOS))
    nome = random.choice(NOMES)
    numero = random.randint(1, 9999)
    cidade_key = random.choice(list(CIDADES))
    comp_key = random.choice(list(COMPLEMENTOS))
    comp_n = random.randint(1, 300)

    # forma canônica
    comp_can, comp_sujo_tpl = random.choice(COMPLEMENTOS[comp_key])
    comp_can = comp_can.format(n=comp_n) if comp_key else ""
    comp_sujo = comp_sujo_tpl.format(n=comp_n) if comp_key else ""
    canonico = f"{tipo} {nome}, {numero}{comp_can} - {cidade_key}"

    # forma suja
    tipo_sujo = random.choice(TIPOS[tipo])
    cidade_suja = random.choice(CIDADES[cidade_key])
    num_sujo = _numero_sujo(numero)
    sep = random.choice([", ", " - ", " ", "; "])
    sujo = f"{tipo_sujo} {nome}{sep}{num_sujo}{comp_sujo}{sep}{cidade_suja}"

    return sujo.strip(), canonico


def main():
    n = 5000
    out = Path("data/raw/addresses.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    linhas = []
    for _ in range(n):
        sujo, canonico = gerar_par()
        linhas.append(f"ENTRADA: {sujo} | SAIDA: {canonico}")

    out.write_text("\n".join(linhas), encoding="utf-8")
    print(f"OK: {n} exemplos salvos em {out}")
    print(f"\nPrimeiros 3 exemplos:")
    for l in linhas[:3]:
        print(" ", l)


if __name__ == "__main__":
    main()
