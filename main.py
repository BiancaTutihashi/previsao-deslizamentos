"""Projeto Mondesa — detecção de ciclos (t0, t1, t2) em séries de potência.

Pipeline organizado pelas fases do roadmap:
  0. primeiro contato com os dados (dados.py)
  1. baseline heurístico (heuristico.py)
  2. rótulos verdadeiros t0/t1/t2 (rotulagem.py)
  3. classificação ponto a ponto: Regressão Logística, Random Forest (modelos.py)
  4. HMM de 2 estados e ruptures (modelos.py)
  5. avaliação por evento: MAE em segundos + precisão/revocação com tolerância (avaliacao.py)
  6. LSTM (lstm.py)
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ciclos import (ciclos_para_estados, estados_para_eventos, estados_para_ciclos,
                    agrupar_ciclos_contiguos, analisar_ciclos, medidas_ciclo)
from dados import secao, carregar_csv, gerar_serie_sintetica, eda, plotar_eda, salvar_eda_html
from heuristico import estimar_limiar_otsu, estados_heuristico
from rotulagem import carregar_rotulos, rotulos_provisorios, salvar_rotulos, plotar_ciclos_rotulados
from avaliacao import avaliar_modelo, metricas_amostra, avaliar_estados
from modelos import ClassificadorCiclos

MODELOS = ['limiar', 'heuristico', 'logreg', 'rf', 'hmm', 'ruptures', 'lstm']
NOMES = {'limiar': 'Limiar simples (Otsu)', 'heuristico': 'Heurístico (mediana+Otsu)',
         'logreg': 'Regressão Logística', 'rf': 'Random Forest', 'hmm': 'HMM 2 estados',
         'ruptures': 'ruptures (KernelCPD)', 'lstm': 'LSTM'}
MIN_DURACAO_S = 1.0   # pulsos mais curtos que isso são ruído (o período ativo dura ~6,5 s)
MARGEM_S = 10.0       # contexto extra em volta de cada trecho avaliado
JANELAS_HEURISTICA = (0.0, 0.15, 0.3)  # filtro de mediana testado na validação (0 = sem filtro)
DEPENDENCIAS = {'hmm': 'hmmlearn', 'ruptures': 'ruptures', 'lstm': 'tensorflow'}


def remover_sem_dependencia(modelos):
    """Pula (com aviso) os modelos cuja biblioteca não está instalada, em vez de quebrar no meio."""
    import importlib.util
    disponiveis = []
    for m in modelos:
        lib = DEPENDENCIAS.get(m)
        if lib and importlib.util.find_spec(lib) is None:
            print(f"⚠️ {NOMES[m]} pulado: a biblioteca '{lib}' não está instalada "
                  f"(no Kaggle: ligue a Internet e rode `!pip install -q {lib}`).")
        else:
            disponiveis.append(m)
    return disponiveis


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv-path', default=os.environ.get(
        'MONDESA_CSV_PATH', '/kaggle/input/datasets/biancatutihashi/experiment4-0/experiment4-0.CSV'))
    p.add_argument('--rotulos', default=os.environ.get('MONDESA_ROTULOS', 'rotulos/rotulos_ciclos.csv'),
                   help='CSV ciclo_id,t0,t1,t2 com os rótulos verdadeiros (Fase 2)')
    p.add_argument('--output-dir', default=os.environ.get('MONDESA_OUTPUT_DIR', './resultados'))
    p.add_argument('--tolerancia', type=float, default=1.0,
                   help='janela de tolerância (s) para contar um evento como acerto')
    p.add_argument('--lacuna', type=float, default=0.5,
                   help='pulsos separados por menos que isso (s) formam um só período ativo '
                        '(rajadas do Wi-Fi); 0,5 s é a regra dos rótulos do rastreamento')
    p.add_argument('--modelos', default=','.join(MODELOS), help=f'subconjunto de {",".join(MODELOS)}')
    p.add_argument('--epocas', type=int, default=100, help='máximo de épocas da LSTM (early stopping)')
    p.add_argument('--lstm-passo', type=int, default=1,
                   help='usa 1 a cada N janelas no treino da LSTM (acelera em CPU)')
    p.add_argument('--sintetico-ciclos', type=int, default=40,
                   help='nº de ciclos da série sintética usada quando o CSV não existe')
    args, _ = p.parse_known_args(argv)
    args.modelos = [m.strip() for m in args.modelos.split(',') if m.strip()]
    desconhecidos = set(args.modelos) - set(MODELOS)
    if desconhecidos:
        p.error(f"modelos desconhecidos: {sorted(desconhecidos)}")
    return args


def obter_rotulos(args, tempo, potencia, ciclos_sinteticos):
    """Fase 2: rótulos manuais se existirem; senão gabarito sintético; senão rótulo provisório."""
    secao("FASE 2: RÓTULOS t0/t1/t2")
    if ciclos_sinteticos is not None:
        print(f"Série sintética: usando os {len(ciclos_sinteticos)} ciclos verdadeiros do gerador.\n")
        return ciclos_sinteticos, 'sintetico'
    if args.rotulos and os.path.exists(args.rotulos):
        ciclos = carregar_rotulos(args.rotulos)
        print(f"✓ {len(ciclos)} ciclos rotulados lidos de {args.rotulos}\n")
        return ciclos, 'arquivo'
    ciclos = rotulos_provisorios(potencia, tempo, limiar_mw=100.0, min_duracao_s=MIN_DURACAO_S)
    caminho = salvar_rotulos(ciclos, os.path.join(args.output_dir, 'rotulos_provisorios.csv'), revisado=0)
    print("⚠️⚠️ SEM RÓTULOS MANUAIS ⚠️⚠️\n"
          f"   '{args.rotulos}' não existe. Usando rótulos PROVISÓRIOS (limiar fixo de 100 mW sobre a\n"
          "   própria potência). Eles NÃO são ground truth: as métricas abaixo medem o quanto os modelos\n"
          "   reproduzem essa regra. Gere rótulos de verdade com `python rotulagem.py propor ...`.\n"
          f"   ({len(ciclos)} ciclos provisórios salvos em {caminho})\n")
    return ciclos, 'provisorio'


def dividir_ciclos(ciclos, frac_treino=0.6, frac_val=0.2):
    """Corte cronológico por ciclo (sem embaralhar): 60% treino, 20% validação, 20% teste."""
    n = len(ciclos)
    if n < 5:
        raise ValueError(f"São necessários pelo menos 5 ciclos rotulados (há {n}).")
    n_tr = max(1, int(round(frac_treino * n)))
    n_va = max(1, int(round(frac_val * n)))
    n_tr = min(n_tr, n - n_va - 1)
    return ciclos[:n_tr], ciclos[n_tr:n_tr + n_va], ciclos[n_tr + n_va:]


def indices_dos_ciclos(tempo, ciclos):
    _, m = ciclos_para_estados(tempo, ciclos)
    return np.flatnonzero(m)


def trechos(tempo, ciclos, margem_s=MARGEM_S):
    """Intervalos de índice [a, b) de cada bloco contínuo de ciclos, com margem de contexto."""
    out = []
    for bloco in agrupar_ciclos_contiguos(ciclos):
        a, b = np.searchsorted(tempo, [bloco[0]['t0'] - margem_s, bloco[-1]['t2'] + margem_s])
        out.append((int(a), int(min(b, len(tempo)))))
    return out


def avaliar_preditor(prever, tempo, estados_reais, ciclos, tol_s, lacuna_s=None, potencia=None,
                     classificador=None):
    """Roda `prever(a, b)` em cada trecho e aplica as métricas de evento (Fase 5).

    Com `classificador` (e `potencia`), cada ciclo detectado também recebe um estado
    (regular/falho) a partir das suas medidas, e entram as métricas de estado.
    """
    subidas, descidas, ciclos_pred, y_real, y_pred = [], [], [], [], []
    _, rotulado = ciclos_para_estados(tempo, ciclos)
    for a, b in trechos(tempo, ciclos):
        est = prever(a, b)
        s, d = estados_para_eventos(est, tempo[a:b], MIN_DURACAO_S, lacuna_s)
        subidas += list(s); descidas += list(d)
        ciclos_pred += estados_para_ciclos(est, tempo[a:b], MIN_DURACAO_S, lacuna_s)
        m = rotulado[a:b]
        y_real.append(estados_reais[a:b][m]); y_pred.append(est[m])
    r = avaliar_modelo(subidas, descidas, ciclos_pred, ciclos, tol_s)
    r.update(metricas_amostra(np.concatenate(y_real), np.concatenate(y_pred)))
    if classificador is not None:
        estados_prev = classificador.prever([medidas_ciclo(c, tempo, potencia) for c in ciclos_pred])
        for c, e in zip(ciclos_pred, estados_prev):
            c['estado'] = e
        r.update(avaliar_estados(ciclos_pred, ciclos, tol_s))
    return r, ciclos_pred


def imprimir_resultado(nome, r, tol_s):
    print(f"{nome}: F1 eventos (±{tol_s}s)={r['f1']*100:.2f}% "
          f"[P={r['precisao']*100:.1f}% R={r['revocacao']*100:.1f}%] | "
          f"MAE t0={r['mae_t0_s']:.3f}s t1={r['mae_t1_s']:.3f}s t2={r['mae_t2_s']:.3f}s | "
          f"ciclos {r['ciclos_recuperados']}/{r['ciclos_reais']}")
    if r.get('falhos_reais'):
        print(f"    regular×falho: acurácia={r['acuracia_estado']*100:.1f}% | falhos: "
              f"{r['falhos_detectados']}/{r['falhos_reais']} detectados, F1 falho={r['f1_falho']*100:.1f}% "
              f"[P={r['precisao_falho']*100:.1f}% R={r['revocacao_falho']*100:.1f}%]")


# ---------- gráficos ----------

def _eixo(ax, titulo, xlabel, ylabel, legenda=True):
    ax.set_title(titulo, fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if legenda:
        ax.legend()


def plotar_historico(history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history['loss'], label='Treino'); a1.plot(history.history['val_loss'], label='Val')
    _eixo(a1, 'Loss (LSTM)', 'Época', 'Loss')
    a2.plot(history.history['accuracy'], label='Treino'); a2.plot(history.history['val_accuracy'], label='Val')
    _eixo(a2, 'Acurácia por amostra (LSTM)', 'Época', 'Acurácia')
    plt.tight_layout(); return fig


def plotar_previsoes_teste(tempo, potencia, ciclos_teste, preditores, margem_s=3.0):
    """Zoom no 1º período ativo do teste: t0/t1 reais + estado previsto por cada modelo.

    O zoom é para enxergar erros da ordem de décimos de segundo; numa janela
    de vários ciclos (~300 s cada) todos os modelos pareceriam idênticos.
    """
    c = ciclos_teste[0]
    a, b = np.searchsorted(tempo, [c['t0'] - margem_s, c['t1'] + margem_s])
    t = tempo[a:b]
    sel = [c]
    fig, axes = plt.subplots(len(preditores) + 1, 1, figsize=(14, 2 + 1.1 * (len(preditores) + 1)),
                             sharex=True, gridspec_kw={'height_ratios': [3] + [1] * len(preditores)})
    axes[0].plot(t, potencia[a:b], '.-', linewidth=0.6, markersize=3, label='Potência')
    for k, cor in (('t0', 'tab:red'), ('t1', 'tab:orange')):
        axes[0].axvline(c[k], color=cor, linestyle='--', linewidth=1)
    axes[0].set_title(f'Teste, ciclo {c.get("ciclo_id", "")}: t0 (vermelho) e t1 (laranja) reais × '
                      'estado previsto por modelo', fontweight='bold')
    axes[0].set_ylabel('mW'); axes[0].grid(True, linestyle='--', alpha=0.4)
    estados_reais, _ = ciclos_para_estados(t, sel)
    for ax, (nome, prever) in zip(axes[1:], preditores.items()):
        ax.fill_between(t, 0, estados_reais, step='post', color='0.85', label='Real')
        ax.step(t, prever(a, b), where='post', linewidth=1.2, label='Previsto')
        ax.set_yticks([0, 1]); ax.set_yticklabels(['DORM', 'ACORD'], fontsize=7)
        ax.set_ylabel(NOMES[nome], rotation=0, ha='right', va='center', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Tempo decorrido (s)')
    plt.tight_layout(); return fig


def plotar_heuristico(tempo, potencia, suave, ciclo):
    """Zoom em um período ativo: sinal bruto × suavizado (mediana não afunda antes do degrau)."""
    a, b = np.searchsorted(tempo, [ciclo['t0'] - 5, ciclo['t1'] + 5])
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(tempo[a:b], potencia[a:b], alpha=0.5, linewidth=0.8, label='Bruta')
    ax.plot(tempo[a:b], suave[a:b], linewidth=2, label='Suavizada (mediana)')
    _eixo(ax, 'Fase 1 — detecção heurística (zoom em um período ativo)', 'Tempo decorrido (s)', 'mW')
    plt.tight_layout(); return fig


def plotar_comparacao(tabela, tol_s):
    """Duas métricas, dois gráficos (sem eixo duplo)."""
    nomes = tabela['modelo'].tolist()
    y = np.arange(len(nomes))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 0.6 * len(nomes) + 2), sharey=True)
    a1.barh(y, tabela['f1'] * 100, height=0.6, color='tab:blue')
    for yi, v in zip(y, tabela['f1'] * 100):
        a1.text(v, yi, f' {v:.1f}%', va='center', fontsize=8)
    a1.set_yticks(y); a1.set_yticklabels(nomes); a1.invert_yaxis()
    a1.set_xlim(0, 110); _eixo(a1, f'F1 de eventos (tolerância ±{tol_s}s)', '%', '', legenda=False)
    h = 0.25
    for i, (k, cor) in enumerate((('mae_t0_s', 'tab:red'), ('mae_t1_s', 'tab:orange'), ('mae_t2_s', 'tab:purple'))):
        a2.barh(y + (i - 1) * h, tabela[k], height=h, color=cor, label=k.split('_')[1])
    _eixo(a2, 'Erro médio absoluto por instante (menor = melhor)', 'segundos', '')
    plt.tight_layout(); return fig


def salvar_graficos(figs, output_dir, formato="pdf", dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for nome, fig in figs.items():
        fig.savefig(os.path.join(output_dir, f"{ts}_{nome}.{formato}"), format=formato, dpi=dpi,
                    bbox_inches='tight')
        plt.close(fig)


# ---------- pipeline ----------

def main(argv=None):
    args = parse_args(argv)
    args.modelos = remover_sem_dependencia(args.modelos)
    os.makedirs(args.output_dir, exist_ok=True)
    tol = args.tolerancia

    # Fase 0
    ciclos_sint = None
    if os.path.exists(args.csv_path):
        df, tempo, potencia = carregar_csv(args.csv_path)
    else:
        print(f"⚠️ {args.csv_path} não encontrado — usando série sintética com gabarito conhecido.")
        df, tempo, potencia, ciclos_sint = gerar_serie_sintetica(n_ciclos=args.sintetico_ciclos)
    eda(df, tempo, potencia)
    figs = {"00_eda": plotar_eda(tempo, potencia)}
    salvar_eda_html(tempo, potencia, os.path.join(args.output_dir, 'html', 'eda_interativo.html'))

    # Fase 2 (vem antes da modelagem: define treino/validação/teste)
    ciclos, origem = obter_rotulos(args, tempo, potencia, ciclos_sint)
    dentro = [c for c in ciclos if c['t0'] >= tempo[0] and c['t2'] <= tempo[-1]]
    if len(dentro) < len(ciclos):
        print(f"⚠️ {len(ciclos) - len(dentro)} ciclos rotulados caem fora dos dados "
              f"({tempo[0]:.0f}–{tempo[-1]:.0f} s) e foram ignorados. O CSV pode estar incompleto.\n")
        ciclos = dentro
    estados, _ = ciclos_para_estados(tempo, ciclos)
    c_tr, c_va, c_te = dividir_ciclos(ciclos)
    idx_tr, idx_va = indices_dos_ciclos(tempo, c_tr), indices_dos_ciclos(tempo, c_va)
    print(f"Ciclos: treino={len(c_tr)} | validação={len(c_va)} | teste={len(c_te)} (corte cronológico)")
    stats = analisar_ciclos([{**c, 'duracao_acordado': c['t1'] - c['t0'], 'duracao_dormindo': c['t2'] - c['t1'],
                              'duracao_ciclo': c['t2'] - c['t0']} for c in ciclos])
    print(f"Rótulos: acordado={stats['duracao_acordado_media']:.2f}s (CV={stats['uniformidade_acordado']:.1f}%) | "
          f"dormindo={stats['duracao_dormindo_media']:.2f}s (CV={stats['uniformidade_dormindo']:.1f}%)\n")

    # Fase 1
    limiar = estimar_limiar_otsu(potencia[idx_tr])
    secao("FASE 1: BASELINES SEM ML")
    print(f"Limiar de Otsu (só potência do treino, sem rótulos): {limiar:.1f} mW")
    preditores = {}

    def f1_val(prever):
        return avaliar_preditor(prever, tempo, estados, c_va, tol, args.lacuna)[0]['f1']

    if 'limiar' in args.modelos:
        bruto = (potencia > limiar).astype(int)
        preditores['limiar'] = lambda a, b: bruto[a:b]
    if 'heuristico' in args.modelos:
        # janela do filtro de mediana escolhida na validação: 0,3 s limpa ruído no exp4,
        # mas apaga os pulsos de ~0,1 s das rajadas do Wi-Fi
        melhor = None
        for janela in JANELAS_HEURISTICA:
            est, suave = estados_heuristico(potencia, tempo, limiar, janela_s=janela,
                                            min_duracao_s=MIN_DURACAO_S, lacuna_s=args.lacuna)
            f1 = f1_val(lambda a, b, est=est: est[a:b])
            print(f"  heurística, janela {janela:.2f} s: F1-val={f1*100:.2f}%")
            if melhor is None or f1 > melhor[0]:
                melhor = (f1, janela, est, suave)
        _, janela, est_heur, suave = melhor
        print(f"✓ Heurística: janela escolhida na validação = {janela:.2f} s")
        preditores['heuristico'] = lambda a, b: est_heur[a:b]
        figs["04_heuristico"] = plotar_heuristico(tempo, potencia, suave, c_te[0])

    # Fase 3
    if {'logreg', 'rf'} & set(args.modelos):
        from modelos import calcular_features, ClassificadorJanela
        secao("FASE 3: CLASSIFICAÇÃO PONTO A PONTO (features de janela)")
        feats = calcular_features(potencia)
        for tipo in ('logreg', 'rf'):
            if tipo in args.modelos:
                t_ini = time.time()
                preditores[tipo] = ClassificadorJanela(tipo).treinar(feats, estados, idx_tr).prever
                print(f"✓ {NOMES[tipo]} treinado em {time.time() - t_ini:.1f}s")

    # Fase 4
    if {'hmm', 'ruptures'} & set(args.modelos):
        from modelos import HMMDoisEstados, Changepoints
        secao("FASE 4: HMM E CHANGEPOINT DETECTION")
        if 'hmm' in args.modelos:
            t_ini = time.time()
            hmm = HMMDoisEstados().treinar(potencia, tempo, estados, idx_tr)
            est_hmm = hmm.prever(0, len(potencia))
            preditores['hmm'] = lambda a, b: est_hmm[a:b]
            print(f"✓ HMM treinado em {time.time() - t_ini:.1f}s "
                  f"(médias dos rótulos: {np.exp(hmm.hmm.means_.ravel()).round(1)} mW)")
        if 'ruptures' in args.modelos:
            cp = Changepoints(limiar).ajustar_penalidade(potencia, idx_tr, f1_val)
            print(f"✓ ruptures: penalidade escolhida na validação = {cp.pen}")
            preditores['ruptures'] = cp.prever

    # Fase 6
    lstm = None
    if 'lstm' in args.modelos:
        from lstm import ModeloLSTM, configurar_sementes
        secao("FASE 6: LSTM")
        configurar_sementes(42)
        lstm = ModeloLSTM(epocas=args.epocas, passo_treino=args.lstm_passo).treinar(potencia, estados, idx_tr, idx_va)
        preditores['lstm'] = lstm.prever
        figs["01_historico_lstm"] = plotar_historico(lstm.history)

    # Fase 5
    secao(f"FASE 5: AVALIAÇÃO POR EVENTO NO TESTE ({len(c_te)} ciclos, tolerância ±{tol}s)")
    linhas = []
    # regular × falho: um só classificador de ciclos, treinado nos ciclos verdadeiros do treino
    classificador = None
    if any(c.get('estado', 'regular') == 'falho' for c in ciclos):
        classificador = ClassificadorCiclos().treinar([medidas_ciclo(c, tempo, potencia) for c in c_tr],
                                                      [c.get('estado', 'regular') for c in c_tr])
        n_f = [sum(c.get('estado') == 'falho' for c in g) for g in (c_tr, c_va, c_te)]
        print(f"Ciclos falhos: treino={n_f[0]} | validação={n_f[1]} | teste={n_f[2]}\n")
    for nome, prever in preditores.items():
        r, _ = avaliar_preditor(prever, tempo, estados, c_te, tol, args.lacuna, potencia, classificador)
        imprimir_resultado(NOMES[nome], r, tol)
        linhas.append({'modelo': NOMES[nome], **r})
    tabela = pd.DataFrame(linhas).sort_values(['f1', 'mae_t0_s'], ascending=[False, True])
    cols = ['modelo', 'f1', 'precisao', 'revocacao', 'f1_subida', 'f1_descida', 'mae_t0_s', 'mae_t1_s',
            'mae_t2_s', 'ciclos_recuperados', 'ciclos_reais', 'acuracia_amostra', 'f1_amostra']
    if classificador is not None:
        cols += ['falhos_reais', 'falhos_detectados', 'acuracia_estado', 'f1_falho', 'precisao_falho', 'revocacao_falho']
    tabela = tabela[cols]
    tabela['rotulos'] = origem
    caminho_tab = os.path.join(args.output_dir, 'comparacao_modelos.csv')
    tabela.to_csv(caminho_tab, index=False, float_format='%.4f')

    secao("COMPARAÇÃO FINAL (ordenada por F1 de eventos)")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(tabela.drop(columns=['rotulos']).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAcurácia por amostra aparece só como referência: ~2% das amostras são ACORDADO,\n"
          f"então ela fica perto de 100% mesmo para modelos que erram os instantes.")
    if origem == 'provisorio':
        print("\n⚠️ Lembrete: rótulos PROVISÓRIOS (limiar 100 mW). Os números acima não valem como resultado.")

    figs["02_previsoes_teste"] = plotar_previsoes_teste(tempo, potencia, c_te, preditores)
    figs["03_comparacao_modelos"] = plotar_comparacao(tabela, tol)
    # conferência: as linhas t0/t1/t2 precisam cair em cima dos degraus de potência
    figs["05_rotulos"] = plotar_ciclos_rotulados(tempo, potencia, ciclos, f"Rótulos ({origem}) — conferência")
    salvar_graficos(figs, os.path.join(args.output_dir, "pdf"))
    print(f"\n✓ Pipeline concluído. Tabela: {caminho_tab} | gráficos: {os.path.join(args.output_dir, 'pdf')}")
    return tabela


if __name__ == "__main__":
    main()
