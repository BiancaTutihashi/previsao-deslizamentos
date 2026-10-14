"""Fase 5 — avaliação por evento (t0/t1/t2), não por amostra.

Acurácia ponto a ponto engana aqui: as transições são raríssimas perto do
total de amostras, então até um modelo ruim passa de 99%. As métricas
principais são:

- precisão / revocação / F1 de eventos com janela de tolerância: um evento
  previsto conta como acerto se ficar a no máximo `tol_s` segundos de um
  evento real ainda não casado (casamento um-para-um);
- MAE em segundos entre o instante previsto e o real, para t0, t1 e t2;
- fração de ciclos reais recuperados (t0, t1 e t2 todos dentro da tolerância).
"""
import numpy as np

from ciclos import agrupar_ciclos_contiguos


def casar_eventos(pred, real, tol_s):
    """Casamento um-para-um, guloso por menor distância. Devolve pares (i_pred, i_real, erro_s)."""
    pred, real = np.asarray(pred, dtype=float), np.asarray(real, dtype=float)
    candidatos = []
    for i, p in enumerate(pred):
        for j in range(np.searchsorted(real, p - tol_s, side='left'), np.searchsorted(real, p + tol_s, side='right')):
            candidatos.append((abs(p - real[j]), i, j))
    candidatos.sort()
    usados_p, usados_r, pares = set(), set(), []
    for d, i, j in candidatos:
        if i not in usados_p and j not in usados_r:
            usados_p.add(i); usados_r.add(j); pares.append((i, j, pred[i] - real[j]))
    return pares


def metricas_eventos(pred, real, tol_s):
    pred, real = np.sort(np.asarray(pred, dtype=float)), np.sort(np.asarray(real, dtype=float))
    pares = casar_eventos(pred, real, tol_s)
    tp = len(pares); fp = len(pred) - tp; fn = len(real) - tp
    precisao = tp / (tp + fp) if (tp + fp) else 0.0
    revocacao = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if (precisao + revocacao) else 0.0
    erros = np.abs([e for _, _, e in pares])
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precisao': precisao, 'revocacao': revocacao, 'f1': f1,
            'mae_s': float(np.mean(erros)) if tp else float('nan'),
            'erro_max_s': float(np.max(erros)) if tp else float('nan')}


def eventos_reais(ciclos):
    """Subidas reais (todos os t0 + o t2 do último ciclo de cada bloco) e descidas (t1)."""
    subidas, descidas = [], []
    for bloco in agrupar_ciclos_contiguos(ciclos):
        subidas += [c['t0'] for c in bloco] + [bloco[-1]['t2']]
        descidas += [c['t1'] for c in bloco]
    return np.array(sorted(subidas)), np.array(sorted(descidas))


def janelas_avaliacao(ciclos, tol_s):
    """Intervalos de tempo cobertos pelos rótulos (eventos previstos fora deles são ignorados)."""
    return [(b[0]['t0'] - tol_s, b[-1]['t2'] + tol_s) for b in agrupar_ciclos_contiguos(ciclos)]


def _dentro(tempos, janelas):
    tempos = np.asarray(tempos, dtype=float)
    m = np.zeros(len(tempos), dtype=bool)
    for a, b in janelas:
        m |= (tempos >= a) & (tempos <= b)
    return tempos[m]


def avaliar_ciclos(ciclos_pred, ciclos_real, tol_s):
    """Para cada ciclo real, procura o previsto com t0 mais próximo e mede o erro de t0/t1/t2."""
    t0_pred = np.array([c['t0'] for c in ciclos_pred])
    erros = {'t0': [], 't1': [], 't2': []}
    recuperados = 0
    for c in ciclos_real:
        if len(t0_pred) == 0:
            break
        j = int(np.argmin(np.abs(t0_pred - c['t0'])))
        if abs(t0_pred[j] - c['t0']) > tol_s:
            continue
        e = {k: abs(ciclos_pred[j][k] - c[k]) for k in ('t0', 't1', 't2')}
        for k in e:
            erros[k].append(e[k])
        recuperados += all(v <= tol_s for v in e.values())
    mae = {f'mae_{k}_s': (float(np.mean(v)) if v else float('nan')) for k, v in erros.items()}
    return {'ciclos_reais': len(ciclos_real), 'ciclos_recuperados': recuperados,
            'frac_ciclos': recuperados / len(ciclos_real) if ciclos_real else 0.0, **mae}


def avaliar_modelo(subidas_pred, descidas_pred, ciclos_pred, ciclos_real, tol_s=1.0):
    """Todas as métricas de evento de um modelo, restritas aos trechos rotulados."""
    janelas = janelas_avaliacao(ciclos_real, tol_s)
    sub_real, desc_real = eventos_reais(ciclos_real)
    sub = metricas_eventos(_dentro(subidas_pred, janelas), sub_real, tol_s)
    desc = metricas_eventos(_dentro(descidas_pred, janelas), desc_real, tol_s)
    tp, fp, fn = sub['tp'] + desc['tp'], sub['fp'] + desc['fp'], sub['fn'] + desc['fn']
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    ciclos_pred = [c for c in ciclos_pred if len(_dentro([c['t0']], janelas))]
    return {'precisao': p, 'revocacao': r, 'f1': 2 * p * r / (p + r) if (p + r) else 0.0,
            'f1_subida': sub['f1'], 'f1_descida': desc['f1'],
            **avaliar_ciclos(ciclos_pred, ciclos_real, tol_s)}


def metricas_amostra(y_real, y_pred):
    """Métrica secundária (por amostra), só para referência."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_real == 1))); fp = int(np.sum((y_pred == 1) & (y_real == 0)))
    fn = int(np.sum((y_pred == 0) & (y_real == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {'acuracia_amostra': float(np.mean(y_real == y_pred)) if len(y_real) else 0.0,
            'f1_amostra': 2 * p * r / (p + r) if (p + r) else 0.0}
