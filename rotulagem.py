"""Fase 2 — rótulos verdadeiros (ground truth) de t0/t1/t2.

Formato do arquivo (CSV): `ciclo_id,t0,t1,t2[,revisado]`, com tempos em
segundos no mesmo eixo de "Tempo decorrido" do pipeline (soma acumulada
de Delta-T). Esse arquivo é usado para treinar E avaliar todos os modelos.

Fluxo de rotulagem manual (não precisa marcar o dataset inteiro):

1. Gerar uma proposta para um trecho, para não começar do zero:

       python rotulagem.py propor --csv-path experiment4-0.CSV --inicio 67500 --fim 71600

   Isso cria `rotulos/rotulos_propostos.csv` (coluna `revisado=0`), um
   gráfico interativo `rotulos/revisao.html` (passe o mouse para ver o
   tempo exato de cada ponto) e um PDF no estilo "um ciclo por linha".
2. Conferir cada ciclo no gráfico, corrigir os tempos no CSV e marcar
   `revisado=1`. Ciclos com `revisado=0` são ignorados pelo pipeline.
3. Repetir para outros trechos e juntar tudo em `rotulos/rotulos_ciclos.csv`.
4. Conferir o resultado final:

       python rotulagem.py visualizar --csv-path experiment4-0.CSV --rotulos rotulos/rotulos_ciclos.csv

Atenção: a proposta vem da heurística da Fase 1. Ela só economiza
digitação — se os ciclos forem aceitos sem conferir, o "gabarito" vira a
própria heurística e a comparação entre modelos volta a ser circular.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ciclos import ciclos_para_estados, estados_para_ciclos

COLUNAS = ['ciclo_id', 't0', 't1', 't2']


def validar_ciclos(ciclos):
    for c in ciclos:
        if not (c['t0'] < c['t1'] < c['t2']):
            raise ValueError(f"Ciclo {c.get('ciclo_id')}: precisa t0 < t1 < t2 (veio {c['t0']}, {c['t1']}, {c['t2']})")
    ordenados = sorted(ciclos, key=lambda c: c['t0'])
    for a, b in zip(ordenados, ordenados[1:]):
        if b['t0'] < a['t2'] - 1e-6:
            raise ValueError(f"Ciclos {a.get('ciclo_id')} e {b.get('ciclo_id')} se sobrepõem")
    return ordenados


ESTADOS = ('regular', 'falho')


def carregar_rotulos(caminho):
    """Lê `ciclo_id,t0,t1,t2[,revisado][,estado]` ou o formato do rastreamento do Pedro (`k,t0,t1,t2,estado`).

    `estado` diz se o ciclo é regular ou falho; sem a coluna, todos são regulares.
    Ciclos ausentes (k pulados) só deixam um buraco: os blocos contínuos são tratados à parte.
    """
    df = pd.read_csv(caminho)
    if 'ciclo_id' not in df.columns and 'k' in df.columns:
        df = df.rename(columns={'k': 'ciclo_id'})
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"{caminho}: faltam as colunas {faltando} (esperado {COLUNAS})")
    if 'revisado' in df.columns:
        n_antes = len(df)
        df = df[df['revisado'].astype(int) == 1]
        if len(df) < n_antes:
            print(f"⚠️ {n_antes - len(df)} ciclos com revisado=0 ignorados em {caminho}")
    estados = df['estado'].astype(str).str.strip().str.lower() if 'estado' in df.columns else pd.Series('regular', index=df.index)
    invalidos = sorted(set(estados) - set(ESTADOS))
    if invalidos:
        raise ValueError(f"{caminho}: estado desconhecido {invalidos} (esperado {ESTADOS})")
    ciclos = [{'ciclo_id': int(r.ciclo_id), 't0': float(r.t0), 't1': float(r.t1), 't2': float(r.t2), 'estado': e}
              for r, e in zip(df.itertuples(), estados)]
    return validar_ciclos(ciclos)


def salvar_rotulos(ciclos, caminho, revisado=None):
    df = pd.DataFrame([{k: c[k] for k in COLUNAS} for c in ciclos], columns=COLUNAS)
    if revisado is not None:
        df['revisado'] = int(revisado)
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    df.to_csv(caminho, index=False, float_format="%.3f")
    return caminho


def rotulos_provisorios(potencia, tempo, limiar_mw=100.0, min_duracao_s=1.0):
    """Ciclos tirados de um limiar fixo sobre a potência bruta (a regra antiga do projeto).

    NÃO é ground truth — é só o que permite rodar o pipeline enquanto não
    há rótulos manuais. Com eles, todos os modelos estão sendo comparados
    com uma regra derivada do próprio sinal.
    """
    estados = (np.asarray(potencia, dtype=float) > limiar_mw).astype(int)
    ciclos = estados_para_ciclos(estados, tempo, min_duracao_s)
    return [{'ciclo_id': i + 1, 't0': c['t0'], 't1': c['t1'], 't2': c['t2']} for i, c in enumerate(ciclos)]


# ---------- gráficos de conferência ----------

CORES = {'t0': 'tab:red', 't1': 'tab:orange', 't2': 'tab:purple'}


def plotar_ciclos_rotulados(tempo, potencia, ciclos, titulo="Rótulos", max_ciclos=13, margem_s=8.0):
    """Visão geral do trecho + um painel por ciclo, com t0/t1/t2 marcados (estilo do coorientador)."""
    ciclos = ciclos[:max_ciclos]
    if not ciclos:
        raise ValueError("Nenhum ciclo para plotar")
    n = len(ciclos)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 2.2 * (n + 1)))
    a, b = ciclos[0]['t0'] - margem_s, ciclos[-1]['t2'] + margem_s
    m = (tempo >= a) & (tempo <= b)
    axes[0].plot(tempo[m], potencia[m], '.', markersize=2)
    for c in ciclos:
        for k in ('t0', 't1'):
            axes[0].axvline(c[k], color=CORES[k], linestyle='--', linewidth=1)
    axes[0].axvline(ciclos[-1]['t2'], color=CORES['t2'], linestyle='--', linewidth=1)
    axes[0].set_title(titulo, fontweight='bold'); axes[0].set_ylabel('mW')

    for ax, c in zip(axes[1:], ciclos):
        m = (tempo >= c['t0'] - 2) & (tempo <= c['t2'] + 2)
        ax.plot(tempo[m], potencia[m], '.', markersize=2, color='tab:blue')
        ax.axvspan(c['t0'], c['t1'], color='tab:orange', alpha=0.15)
        for k in ('t0', 't1', 't2'):
            ax.axvline(c[k], color=CORES[k], linestyle='--', linewidth=1.2, label=k)
        ax.set_title(f"C{c.get('ciclo_id', '')}: t0={c['t0']:.3f}s | t1={c['t1']:.3f}s | t2={c['t2']:.3f}s",
                     fontsize=9)
        ax.set_ylabel('mW')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Tempo decorrido (s)')
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig


def salvar_revisao_html(tempo, potencia, ciclos, caminho, margem_s=8.0):
    """Gráfico interativo com os pontos brutos e as linhas t0/t1/t2, para conferir tempos exatos."""
    import plotly.graph_objects as go
    a, b = ciclos[0]['t0'] - margem_s, ciclos[-1]['t2'] + margem_s
    m = (tempo >= a) & (tempo <= b)
    fig = go.Figure(go.Scattergl(x=tempo[m], y=potencia[m], mode='markers', marker=dict(size=3),
                                 name='Pontos reais', hovertemplate='t=%{x:.3f}s<br>%{y:.1f} mW'))
    for c in ciclos:
        for k in ('t0', 't1', 't2'):
            fig.add_vline(x=c[k], line_dash='dash', line_color=CORES[k].replace('tab:', ''),
                          annotation_text=f"C{c['ciclo_id']} {k}" if k != 't2' else None)
    fig.update_layout(title='Revisão dos rótulos (zoom com o mouse; hover mostra o tempo exato)',
                      xaxis_title='Tempo decorrido (s)', yaxis_title='mW')
    os.makedirs(os.path.dirname(caminho) or '.', exist_ok=True)
    fig.write_html(caminho, include_plotlyjs='cdn')
    return caminho


# ---------- linha de comando ----------

def _carregar_serie(args):
    from dados import carregar_csv, gerar_serie_sintetica
    if args.csv_path and os.path.exists(args.csv_path):
        _, tempo, potencia = carregar_csv(args.csv_path)
    else:
        print("⚠️ CSV não encontrado — usando série sintética.")
        _, tempo, potencia, _ = gerar_serie_sintetica()
    return tempo, potencia


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('propor', help='propõe ciclos num trecho para revisão manual')
    pv = sub.add_parser('visualizar', help='plota um arquivo de rótulos sobre o sinal')
    for s in (pp, pv):
        s.add_argument('--csv-path', default=os.environ.get('MONDESA_CSV_PATH'))
        s.add_argument('--saida-dir', default='rotulos')
    pp.add_argument('--inicio', type=float, default=None, help='início do trecho (s)')
    pp.add_argument('--fim', type=float, default=None, help='fim do trecho (s)')
    pv.add_argument('--rotulos', required=True)
    args = p.parse_args()

    tempo, potencia = _carregar_serie(args)
    if args.cmd == 'propor':
        from heuristico import estimar_limiar_otsu, detectar_ciclos_heuristico
        ini = tempo[0] if args.inicio is None else args.inicio
        fim = min(tempo[-1], ini + 4000) if args.fim is None else args.fim
        m = (tempo >= ini) & (tempo <= fim)
        ciclos, _ = detectar_ciclos_heuristico(potencia[m], tempo[m], estimar_limiar_otsu(potencia))
        ciclos = [{'ciclo_id': i + 1, **{k: c[k] for k in ('t0', 't1', 't2')}} for i, c in enumerate(ciclos)]
        if not ciclos:
            raise SystemExit("Nenhum ciclo encontrado no trecho.")
        csv = salvar_rotulos(ciclos, os.path.join(args.saida_dir, 'rotulos_propostos.csv'), revisado=0)
        html = salvar_revisao_html(tempo, potencia, ciclos, os.path.join(args.saida_dir, 'revisao.html'))
        fig = plotar_ciclos_rotulados(tempo, potencia, ciclos, f"Proposta {ini:.0f}–{fim:.0f} s (REVISAR)")
        pdf = os.path.join(args.saida_dir, 'revisao.pdf'); fig.savefig(pdf, bbox_inches='tight')
        print(f"✓ {len(ciclos)} ciclos propostos.\n  CSV:  {csv}\n  HTML: {html}\n  PDF:  {pdf}\n"
              "Corrija os tempos no CSV, marque revisado=1 e salve como rotulos/rotulos_ciclos.csv.")
    else:
        ciclos = carregar_rotulos(args.rotulos)
        _, rotulado = ciclos_para_estados(tempo, ciclos)
        print(f"✓ {len(ciclos)} ciclos válidos cobrindo {rotulado.mean()*100:.1f}% das amostras.")
        fig = plotar_ciclos_rotulados(tempo, potencia, ciclos, f"Rótulos: {os.path.basename(args.rotulos)}")
        pdf = os.path.join(args.saida_dir, 'rotulos_conferencia.pdf')
        os.makedirs(args.saida_dir, exist_ok=True); fig.savefig(pdf, bbox_inches='tight')
        print(f"  PDF: {pdf}")


if __name__ == '__main__':
    main()
