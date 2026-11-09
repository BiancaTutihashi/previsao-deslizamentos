"""Utilitários comuns: estados ponto a ponto <-> transições <-> ciclos (t0, t1, t2).

Convenções usadas em todo o projeto:
- estado 1 = ACORDADO (período ativo), 0 = DORMINDO;
- t0 = instante em que o ciclo começa (DORMINDO -> ACORDADO);
- t1 = fim do período ativo (ACORDADO -> DORMINDO);
- t2 = fim do período dormindo = t0 do ciclo seguinte;
- t0/t2 = tempo da primeira amostra ACORDADO; t1 = tempo da última amostra
  ACORDADO (mesma convenção dos rótulos do coorientador).
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


def limpar_estados(estados, tempo, lacuna_s=0.5, min_ativo_s=1.0):
    """Junta pulsos separados por menos de `lacuna_s` e depois descarta os mais curtos que `min_ativo_s`.

    A ordem importa: no Wi-Fi o período ativo é uma rajada de pulsos de ~0,1 s.
    Descartando os curtos primeiro (como faz `aplicar_debouncing_estados`), a
    rajada inteira some; juntando primeiro, ela vira um período ativo só.
    A lacuna é medida da última amostra de um pulso à primeira do seguinte, e
    a duração de um período ativo de t0 a t1 (primeira e última amostra acordada).
    """
    e = np.asarray(estados).astype(int)
    tempo = np.asarray(tempo, dtype=float)
    borda = np.diff(np.concatenate(([0], e, [0])))
    ini, fim = np.flatnonzero(borda == 1), np.flatnonzero(borda == -1) - 1  # fim = última amostra acordada
    limpo = np.zeros(len(e), dtype=int)
    if len(ini) == 0:
        return limpo
    junta = tempo[ini[1:]] - tempo[fim[:-1]] < lacuna_s
    blocos_ini = np.concatenate(([ini[0]], ini[1:][~junta]))
    blocos_fim = np.concatenate((fim[:-1][~junta], [fim[-1]]))
    for a, b in zip(blocos_ini, blocos_fim):
        if tempo[b] - tempo[a] >= min_ativo_s:
            limpo[a:b + 1] = 1
    return limpo


def detectar_transicoes(previsoes, tempo, min_duracao=1.0, verbose=True, lacuna_s=None):
    """Índices, tempos e estado novo de cada transição, depois da limpeza dos pulsos.

    Com `lacuna_s` usa `limpar_estados` (junta pulsos próximos e descarta períodos
    ativos curtos); sem ele, o debouncing antigo.
    """
    if lacuna_s is None:
        suave = aplicar_debouncing_estados(previsoes, tempo, min_duracao)
    else:
        suave = limpar_estados(previsoes, tempo, lacuna_s, min_duracao)
    idx = np.where(np.diff(suave) != 0)[0] + 1
    if verbose:
        print(f"✓ {len(idx)} transições detectadas." if len(idx) else "⚠️ Nenhuma transição detectada.")
    if len(idx) == 0:
        return np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    novos = suave[idx]
    # subida: primeira amostra acordada; descida (t1): última amostra acordada
    tempos = np.asarray(tempo)[np.where(novos == 1, idx, idx - 1)]
    return idx, tempos, novos


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


def estados_para_eventos(estados, tempo, min_duracao=1.0, lacuna_s=None):
    """Tempos das subidas (t0/t2) e das descidas (t1) de uma série de estados."""
    _, tempos, novos = detectar_transicoes(estados, tempo, min_duracao, verbose=False, lacuna_s=lacuna_s)
    return tempos[novos == 1], tempos[novos == 0]


def estados_para_ciclos(estados, tempo, min_duracao=1.0, lacuna_s=None):
    _, tempos, novos = detectar_transicoes(estados, tempo, min_duracao, verbose=False, lacuna_s=lacuna_s)
    return extrair_ciclos(tempos, novos, verbose=False)


def ciclos_para_estados(tempo, ciclos):
    """Rótulo ponto a ponto a partir de ciclos (1 em [t0, t1]) e máscara de amostras rotuladas.

    Só as amostras dentro de algum ciclo [t0, t2) são consideradas rotuladas:
    a rotulagem manual cobre um subconjunto do dataset.
    """
    tempo = np.asarray(tempo, dtype=float)
    estados = np.zeros(len(tempo), dtype=int)
    rotulado = np.zeros(len(tempo), dtype=bool)
    for c in ciclos:
        i0, i2 = np.searchsorted(tempo, [c['t0'], c['t2']])
        i1 = np.searchsorted(tempo, c['t1'], side='right')  # t1 é a última amostra acordada
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


MEDIDAS = ('T_on', 'T', 'P_on', 'P_off', 'P_max')


def medidas_ciclo(ciclo, tempo, potencia):
    """Medidas de um ciclo (as mesmas grandezas do critério de falho do coorientador, e mais duas).

    T_on = t1 − t0; T = t2 − t0; P_on = potência média em [t0, t1];
    P_off = potência média em (t1, t2); P_max = pico em [t0, t1].
    """
    tempo = np.asarray(tempo, dtype=float)
    i0, i2 = np.searchsorted(tempo, [ciclo['t0'], ciclo['t2']])
    i1 = np.searchsorted(tempo, ciclo['t1'], side='right')  # t1 é a última amostra acordada
    on, off = potencia[i0:max(i1, i0 + 1)], potencia[i1:max(i2, i1 + 1)]
    return {'T_on': ciclo['t1'] - ciclo['t0'], 'T': ciclo['t2'] - ciclo['t0'],
            'P_on': float(np.mean(on)), 'P_off': float(np.mean(off)) if len(off) else float('nan'),
            'P_max': float(np.max(on))}
