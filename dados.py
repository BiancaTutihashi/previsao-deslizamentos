"""Fase 0 — carregamento, primeiro contato com os dados e série sintética de teste."""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ciclos import _calcular_cv_seguro


def secao(titulo):
    print("=" * 80); print(titulo); print("=" * 80)


def carregar_csv(path):
    """Lê o CSV do experimento e devolve (df, tempo_s, potencia_mW)."""
    df = pd.read_csv(path)
    if "Delta-T_[ms]" in df.columns:
        tempo = df["Delta-T_[ms]"].cumsum().to_numpy(dtype=float) / 1000.0
    else:
        tempo = np.arange(len(df), dtype=float)
    potencia = df["Consumed_[mW]"].to_numpy(dtype=float)
    return df, tempo, potencia


def gerar_serie_sintetica(n_ciclos=40, periodo_s=306.1, ativo_s=6.6, dt_s=0.057, seed=0):
    """Série parecida com o experiment4-0, com os t0/t1/t2 verdadeiros conhecidos.

    Dormindo ≈ 55 mW com picos isolados de ruído; acordado = rajada de
    300–700 mW seguida de platô ≈ 160 mW. Serve para testar o pipeline
    quando o CSV real não está disponível — aqui o gabarito é independente
    do sinal, então as métricas são de verdade.
    """
    rng = np.random.default_rng(seed)
    duracao = (n_ciclos + 1) * periodo_s + 20.0
    tempo = np.cumsum(rng.normal(dt_s, dt_s * 0.05, int(duracao / dt_s)))
    potencia = rng.normal(55, 2, len(tempo))
    picos = rng.random(len(tempo)) < 0.003
    potencia[picos] = rng.uniform(65, 100, picos.sum())

    # gera um período ativo a mais: o t2 do último ciclo precisa de uma subida depois dele
    ciclos, t = [], 10.0
    for k in range(n_ciclos + 1):
        ativo = ativo_s + rng.normal(0, 0.1)
        t0, t1 = t, t + ativo
        t2 = t0 + periodo_s + rng.normal(0, 0.05)
        i0, i1 = np.searchsorted(tempo, [t0, t1])
        n_rajada = max(2, int(0.4 / dt_s))
        potencia[i0:i1] = rng.normal(160, 6, i1 - i0)
        potencia[i0:i0 + n_rajada] = rng.uniform(250, 700, n_rajada)
        # gabarito: t0 = 1ª amostra acordada, t1 = última amostra acordada (convenção do projeto)
        ciclos.append({'ciclo_id': k + 1, 't0': float(tempo[i0]), 't1': float(tempo[i1 - 1]),
                       't2': float(tempo[np.searchsorted(tempo, t2)])})
        t = t2
    ciclos = ciclos[:n_ciclos]
    df = pd.DataFrame({"Delta-T_[ms]": np.diff(tempo, prepend=0.0) * 1000.0, "Consumed_[mW]": potencia})
    return df, tempo, potencia, ciclos


def eda(df, tempo, potencia):
    """Resumo textual do CSV (Fase 0 do roadmap)."""
    secao("FASE 0: PRIMEIRO CONTATO COM OS DADOS")
    print(f"Linhas: {len(df)} | Colunas: {list(df.columns)}")
    print(f"Tipos: {dict(df.dtypes.astype(str))} | Nulos: {int(df.isnull().sum().sum())}")
    diffs = np.diff(tempo)
    cv = _calcular_cv_seguro(diffs)
    status = "⚠️ irregular" if cv > 10 else "✓ regular"
    print(f"Duração: {tempo[-1] - tempo[0]:.1f} s | passo médio={np.mean(diffs)*1000:.1f} ms "
          f"(CV={cv:.2f}% {status})")
    print(f"Potência: min={potencia.min():.1f} mediana={np.median(potencia):.1f} max={potencia.max():.1f} mW\n")


def _decimar_min_max(tempo, potencia, max_pontos=200_000):
    """Reduz a série para plot preservando picos (min e max de cada bloco)."""
    n = len(tempo)
    if n <= max_pontos:
        return tempo, potencia
    passo = int(np.ceil(n / (max_pontos / 2)))
    n_corte = (n // passo) * passo
    p = potencia[:n_corte].reshape(-1, passo)
    t = tempo[:n_corte].reshape(-1, passo)
    imin, imax = p.argmin(axis=1), p.argmax(axis=1)
    linhas = np.arange(len(p))
    ordem = np.sort(np.stack([imin, imax], axis=1), axis=1)
    return t[linhas[:, None], ordem].ravel(), p[linhas[:, None], ordem].ravel()


def plotar_eda(tempo, potencia, n_ciclos_zoom=3, periodo_estimado_s=306.0):
    """Série inteira + recorte de poucos ciclos, para enxergar o formato do degrau."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7))
    t_dec, p_dec = _decimar_min_max(tempo, potencia, 50_000)
    a1.plot(t_dec, p_dec, linewidth=0.6)
    a1.set_title("Série completa", fontweight="bold"); a1.set_ylabel("mW")
    fim = tempo[0] + n_ciclos_zoom * periodo_estimado_s
    m = tempo <= fim
    a2.plot(tempo[m], potencia[m], linewidth=0.8)
    a2.set_title(f"Recorte (~{n_ciclos_zoom} ciclos)", fontweight="bold")
    a2.set_xlabel("Tempo decorrido (s)"); a2.set_ylabel("mW")
    for ax in (a1, a2):
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def salvar_eda_html(tempo, potencia, caminho):
    """Gráfico interativo (zoom) da série inteira com plotly, se instalado."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️ plotly não instalado — pulando o gráfico interativo.")
        return None
    t_dec, p_dec = _decimar_min_max(tempo, potencia)
    fig = go.Figure(go.Scattergl(x=t_dec, y=p_dec, mode="lines", line=dict(width=1), name="Potência"))
    fig.update_layout(title="Potência × tempo (zoom com o mouse)",
                      xaxis_title="Tempo decorrido (s)", yaxis_title="mW")
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    fig.write_html(caminho, include_plotlyjs="cdn")
    return caminho
