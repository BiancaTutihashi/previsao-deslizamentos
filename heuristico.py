"""Fase 1 — baseline heurístico (regra fixa, sem ML).

Regra: suaviza o sinal com filtro de mediana, marca ACORDADO onde a
potência suavizada passa do limiar e descarta pulsos mais curtos que
`min_duracao_s`. O limiar vem do método de Otsu, estimado só com a
potência (sem olhar rótulo nenhum).

Por que mediana e não Savitzky-Golay: o Savitzky-Golay ajusta polinômios
e, num degrau, "afunda" antes da subida (overshoot/ringing) — dava para
ver a curva suavizada indo a ~25 mW antes de cada pico. A mediana remove
os picos isolados de ruído sem deslocar nem deformar as bordas do degrau.
"""
import numpy as np
from scipy.signal import medfilt

from ciclos import aplicar_debouncing_estados, estados_para_ciclos, limpar_estados


def estimar_limiar_otsu(potencia, bins=256):
    """Limiar não supervisionado (Otsu) estimado só com a potência, sem ver os rótulos."""
    hist, bordas = np.histogram(np.asarray(potencia, dtype=float), bins=bins)
    centros = (bordas[:-1] + bordas[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    soma0 = np.cumsum(hist * centros); soma1 = soma0[-1] - soma0
    with np.errstate(divide='ignore', invalid='ignore'):
        var_entre = w0 * w1 * (soma0 / w0 - soma1 / w1) ** 2
    var_entre = np.nan_to_num(var_entre[:-1], nan=-1.0)
    # entre modos bem separados há um platô de máximo (bins vazios): usa o meio dele
    k = np.flatnonzero(np.isclose(var_entre, var_entre.max()))
    k = k[k - k[0] == np.arange(len(k))]  # só o primeiro bloco contíguo
    return float(bordas[(k[0] + k[-1]) // 2 + 1])


def suavizar(potencia, tempo, janela_s=0.3):
    """Filtro de mediana com janela ~janela_s segundos (nº ímpar de amostras); janela_s <= 0 = sem filtro."""
    if janela_s <= 0:
        return np.asarray(potencia, dtype=float).copy()
    dt = float(np.median(np.diff(tempo))) if len(tempo) > 1 else 1.0
    k = max(3, int(round(janela_s / dt)) | 1)
    k = min(k, len(potencia) - (1 - len(potencia) % 2))
    return medfilt(np.asarray(potencia, dtype=float), kernel_size=max(1, k))


def estados_heuristico(potencia, tempo, limiar, janela_s=0.3, min_duracao_s=1.0, lacuna_s=None):
    """Estado ponto a ponto (0/1) segundo a regra heurística.

    Com `lacuna_s`, pulsos separados por menos que isso viram um período ativo só
    (rajadas do Wi-Fi) antes do descarte dos períodos curtos; sem ele, o debouncing antigo.
    """
    suave = suavizar(potencia, tempo, janela_s)
    estados = (suave > limiar).astype(int)
    if lacuna_s is None:
        return aplicar_debouncing_estados(estados, tempo, min_duracao_s), suave
    return limpar_estados(estados, tempo, lacuna_s, min_duracao_s), suave


def detectar_ciclos_heuristico(potencia, tempo, limiar, janela_s=0.3, min_duracao_s=1.0, lacuna_s=None):
    """Recebe a série e devolve a lista de ciclos (t0, t1, t2) detectados + sinal suavizado."""
    if len(potencia) <= 3:
        raise ValueError("Sinal muito curto")
    estados, suave = estados_heuristico(potencia, tempo, limiar, janela_s, min_duracao_s, lacuna_s)
    return estados_para_ciclos(estados, tempo, min_duracao_s, lacuna_s), suave
