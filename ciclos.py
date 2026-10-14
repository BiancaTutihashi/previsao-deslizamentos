"""Utilitários comuns: estados ponto a ponto <-> transições <-> ciclos (t0, t1, t2).

Convenções usadas em todo o projeto:
- estado 1 = ACORDADO (período ativo), 0 = DORMINDO;
- t0 = instante em que o ciclo começa (DORMINDO -> ACORDADO);
- t1 = fim do período ativo (ACORDADO -> DORMINDO);
- t2 = fim do período dormindo = t0 do ciclo seguinte;
- o tempo de uma transição é o da primeira amostra no estado novo.
"""
import numpy as np


def _calcular_cv_seguro(vetor):
    if len(vetor) == 0:
        return 0.0
    media, desvio = float(np.mean(vetor)), float(np.std(vetor))
    if media == 0 or np.isnan(media) or np.isnan(desvio):
        return 0.0
    return (desvio / media) * 100.0


def aplicar_debouncing_estados(previsoes, tempo, min_duracao=1.0):
    """Rederiva os estados após filtrar pulsos curtos, preservando alternância estrita."""
    prev = np.array(previsoes, copy=True).astype(int)
    n = len(prev)
    if n == 0:
        return prev
    idx0, estado = 0, prev[0]
    for i in range(1, n):
        if prev[i] != estado:
            if tempo[i] - tempo[idx0] < min_duracao:
                prev[idx0:i] = 1 - estado
            estado, idx0 = prev[i], i
    if tempo[-1] - tempo[idx0] < min_duracao:
        prev[idx0:] = prev[max(0, idx0 - 1)]
    return prev


def detectar_transicoes(previsoes, tempo, min_duracao=1.0, verbose=True):
    """Índices, tempos e estado novo de cada transição (após debouncing)."""
    suave = aplicar_debouncing_estados(previsoes, tempo, min_duracao)
    idx = np.where(np.diff(suave) != 0)[0] + 1
    if verbose:
        print(f"✓ {len(idx)} transições detectadas." if len(idx) else "⚠️ Nenhuma transição detectada.")
    if len(idx) == 0:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    return idx, np.asarray(tempo)[idx], suave[idx]


def extrair_ciclos(tempos, estados, verbose=True):
    """Agrupa transições subida/descida/subida em ciclos (t0, t1, t2)."""
    ciclos, i = [], 0
    while i + 2 < len(tempos):
        if estados[i] == 1 and estados[i + 1] == 0 and estados[i + 2] == 1:
            ciclos.append({
                'ciclo_num': len(ciclos) + 1,
                't0': float(tempos[i]), 't1': float(tempos[i + 1]), 't2': float(tempos[i + 2]),
                'duracao_acordado': float(tempos[i + 1] - tempos[i]),
                'duracao_dormindo': float(tempos[i + 2] - tempos[i + 1]),
                'duracao_ciclo': float(tempos[i + 2] - tempos[i]),
            })
            i += 2
        else:
            i += 1
    if verbose:
        print(f"✓ {len(ciclos)} ciclos extraídos.\n")
    return ciclos


def estados_para_eventos(estados, tempo, min_duracao=1.0):
    """Tempos das subidas (t0/t2) e das descidas (t1) de uma série de estados."""
    _, tempos, novos = detectar_transicoes(estados, tempo, min_duracao, verbose=False)
    return tempos[novos == 1], tempos[novos == 0]


def estados_para_ciclos(estados, tempo, min_duracao=1.0):
    _, tempos, novos = detectar_transicoes(estados, tempo, min_duracao, verbose=False)
    return extrair_ciclos(tempos, novos, verbose=False)


def ciclos_para_estados(tempo, ciclos):
    """Rótulo ponto a ponto a partir de ciclos (1 em [t0, t1)) e máscara de amostras rotuladas.

    Só as amostras dentro de algum ciclo [t0, t2) são consideradas rotuladas:
    a rotulagem manual cobre um subconjunto do dataset.
    """
    tempo = np.asarray(tempo, dtype=float)
    estados = np.zeros(len(tempo), dtype=int)
    rotulado = np.zeros(len(tempo), dtype=bool)
    for c in ciclos:
        i0, i1, i2 = np.searchsorted(tempo, [c['t0'], c['t1'], c['t2']])
        estados[i0:i1] = 1
        rotulado[i0:i2] = True
    return estados, rotulado


def analisar_ciclos(ciclos):
    if not ciclos:
        return None
    ac = np.array([c['duracao_acordado'] for c in ciclos])
    do = np.array([c['duracao_dormindo'] for c in ciclos])
    ci = np.array([c['duracao_ciclo'] for c in ciclos])
    return {'num_ciclos': len(ciclos),
            'duracao_acordado_media': float(np.mean(ac)), 'uniformidade_acordado': _calcular_cv_seguro(ac),
            'duracao_dormindo_media': float(np.mean(do)), 'uniformidade_dormindo': _calcular_cv_seguro(do),
            'duracao_ciclo_media': float(np.mean(ci)), 'uniformidade_ciclo': _calcular_cv_seguro(ci)}


def agrupar_ciclos_contiguos(ciclos, tol=1e-6):
    """Separa ciclos em blocos onde t2 de um == t0 do seguinte (trechos rotulados contínuos)."""
    blocos = []
    for c in sorted(ciclos, key=lambda c: c['t0']):
        if blocos and abs(blocos[-1][-1]['t2'] - c['t0']) <= tol:
            blocos[-1].append(c)
        else:
            blocos.append([c])
    return blocos
