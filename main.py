"""Projeto Mondesa — detecção de ciclos (t0, t1, t2) em séries de potência. Versão ENXUTA para o Kaggle.

Gerado por gerar_arquivo_unico.py a partir dos módulos do repositório: para mudar o
código, edite os módulos e gere de novo. Rótulos do chunk 12 embutidos (ROTULOS_EMBUTIDOS).

    !pip install -q hmmlearn ruptures
    !python mondesa_kaggle_enxuto.py --csv-path /kaggle/input/.../experiment4-0.CSV --output-dir /kaggle/working/resultados
"""
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import argparse
import time
from datetime import datetime
import io


# ============================== ciclos.py ==============================
def _calcular_cv_seguro(vetor):
    if len(vetor) == 0:
        return 0.0
    media, desvio = float(np.mean(vetor)), float(np.std(vetor))
    if media == 0 or np.isnan(media) or np.isnan(desvio):
        return 0.0
    return (desvio / media) * 100.0


def aplicar_debouncing_estados(previsoes, tempo, min_duracao=1.0):
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
    suave = aplicar_debouncing_estados(previsoes, tempo, min_duracao)
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
    _, tempos, novos = detectar_transicoes(estados, tempo, min_duracao, verbose=False)
    return tempos[novos == 1], tempos[novos == 0]


def estados_para_ciclos(estados, tempo, min_duracao=1.0):
    _, tempos, novos = detectar_transicoes(estados, tempo, min_duracao, verbose=False)
    return extrair_ciclos(tempos, novos, verbose=False)


def ciclos_para_estados(tempo, ciclos):
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
    blocos = []
    for c in sorted(ciclos, key=lambda c: c['t0']):
        if blocos and abs(blocos[-1][-1]['t2'] - c['t0']) <= tol:
            blocos[-1].append(c)
        else:
            blocos.append([c])
    return blocos



# ============================== dados.py ==============================
def secao(titulo):
    print("=" * 80); print(titulo); print("=" * 80)


def carregar_csv(path):
    df = pd.read_csv(path)
    if "Delta-T_[ms]" in df.columns:
        tempo = df["Delta-T_[ms]"].cumsum().to_numpy(dtype=float) / 1000.0
    else:
        tempo = np.arange(len(df), dtype=float)
    potencia = df["Consumed_[mW]"].to_numpy(dtype=float)
    return df, tempo, potencia


def eda(df, tempo, potencia):
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



# ============================== heuristico.py ==============================
def estimar_limiar_otsu(potencia, bins=256):
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
    dt = float(np.median(np.diff(tempo))) if len(tempo) > 1 else 1.0
    k = max(3, int(round(janela_s / dt)) | 1)
    k = min(k, len(potencia) - (1 - len(potencia) % 2))
    return medfilt(np.asarray(potencia, dtype=float), kernel_size=max(1, k))


def estados_heuristico(potencia, tempo, limiar, janela_s=0.3, min_duracao_s=1.0):
    suave = suavizar(potencia, tempo, janela_s)
    estados = (suave > limiar).astype(int)
    return aplicar_debouncing_estados(estados, tempo, min_duracao_s), suave


def detectar_ciclos_heuristico(potencia, tempo, limiar, janela_s=0.3, min_duracao_s=1.0):
    if len(potencia) <= 3:
        raise ValueError("Sinal muito curto")
    estados, suave = estados_heuristico(potencia, tempo, limiar, janela_s, min_duracao_s)
    return estados_para_ciclos(estados, tempo, min_duracao_s), suave



# ============================== avaliacao.py ==============================
def casar_eventos(pred, real, tol_s):
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
    subidas, descidas = [], []
    for bloco in agrupar_ciclos_contiguos(ciclos):
        subidas += [c['t0'] for c in bloco] + [bloco[-1]['t2']]
        descidas += [c['t1'] for c in bloco]
    return np.array(sorted(subidas)), np.array(sorted(descidas))


def janelas_avaliacao(ciclos, tol_s):
    return [(b[0]['t0'] - tol_s, b[-1]['t2'] + tol_s) for b in agrupar_ciclos_contiguos(ciclos)]


def _dentro(tempos, janelas):
    tempos = np.asarray(tempos, dtype=float)
    m = np.zeros(len(tempos), dtype=bool)
    for a, b in janelas:
        m |= (tempos >= a) & (tempos <= b)
    return tempos[m]


def avaliar_ciclos(ciclos_pred, ciclos_real, tol_s):
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
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_real == 1))); fp = int(np.sum((y_pred == 1) & (y_real == 0)))
    fn = int(np.sum((y_pred == 0) & (y_real == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {'acuracia_amostra': float(np.mean(y_real == y_pred)) if len(y_real) else 0.0,
            'f1_amostra': 2 * p * r / (p + r) if (p + r) else 0.0}



# ============================== rotulagem.py ==============================
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


def carregar_rotulos(caminho):
    df = pd.read_csv(caminho)
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"{caminho}: faltam as colunas {faltando} (esperado {COLUNAS})")
    if 'revisado' in df.columns:
        n_antes = len(df)
        df = df[df['revisado'].astype(int) == 1]
        if len(df) < n_antes:
            print(f"⚠️ {n_antes - len(df)} ciclos com revisado=0 ignorados em {caminho}")
    ciclos = [{'ciclo_id': int(r.ciclo_id), 't0': float(r.t0), 't1': float(r.t1), 't2': float(r.t2)}
              for r in df.itertuples()]
    return validar_ciclos(ciclos)


CORES = {'t0': 'tab:red', 't1': 'tab:orange', 't2': 'tab:purple'}


def plotar_ciclos_rotulados(tempo, potencia, ciclos, titulo="Rótulos", max_ciclos=13, margem_s=8.0):
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



# ============================== modelos.py ==============================
JANELAS = (5, 20, 60)  # em amostras (~0,3 s, ~1,1 s, ~3,4 s com passo de 57 ms)


def calcular_features(potencia):
    p = pd.Series(np.asarray(potencia, dtype=float))
    f = {'p': p, 'log_p': np.log(p.clip(lower=1e-3)), 'diff1': p.diff(1), 'diff5': p.diff(5)}
    for w in JANELAS:
        r = p.rolling(w, min_periods=1)
        f[f'media_{w}'] = r.mean(); f[f'mediana_{w}'] = r.median()
        f[f'std_{w}'] = r.std(); f[f'min_{w}'] = r.min(); f[f'max_{w}'] = r.max()
    return pd.DataFrame(f).fillna(0.0).to_numpy(dtype=np.float32)


def _amostrar_treino(idx, y, max_amostras, seed=0):
    if len(idx) <= max_amostras:
        return idx
    rng = np.random.default_rng(seed)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = rng.choice(neg, size=max(0, max_amostras - len(pos)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


class ClassificadorJanela:

    def __init__(self, tipo='rf', max_amostras=300_000, seed=42):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        self.max_amostras, self.seed = max_amostras, seed
        if tipo == 'logreg':
            self.modelo = make_pipeline(StandardScaler(),
                                        LogisticRegression(class_weight='balanced', max_iter=2000))
        elif tipo == 'rf':
            self.modelo = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                                 class_weight='balanced', n_jobs=-1, random_state=seed)
        else:
            raise ValueError(tipo)

    def treinar(self, features, estados, idx_treino):
        idx = _amostrar_treino(idx_treino, estados, self.max_amostras, self.seed)
        self.features = features
        self.modelo.fit(features[idx], estados[idx])
        return self

    def prever(self, a, b):
        return self.modelo.predict(self.features[a:b]).astype(int)


class HMMDoisEstados:

    def __init__(self, var_min=0.01):
        self.var_min = var_min  # em log(mW)²: desvio de ~10% da potência

    def treinar(self, potencia, tempo, estados, idx_treino):
        from hmmlearn.hmm import GaussianHMM
        x = np.log(np.clip(suavizar(potencia, tempo), 1e-3, None)).reshape(-1, 1)
        self.x = x
        xt, yt = x[idx_treino], estados[idx_treino]
        hmm = GaussianHMM(n_components=2, covariance_type='diag', init_params='', params='')
        hmm.n_features = 1
        hmm.startprob_ = np.array([0.5, 0.5])
        p_sai = 1.0 / max(1, np.mean(yt == 0) * len(yt) / max(1, np.sum(np.diff(yt) == 1)))
        p_volta = 1.0 / max(1, np.mean(yt == 1) * len(yt) / max(1, np.sum(np.diff(yt) == -1)))
        hmm.transmat_ = np.array([[1 - p_sai, p_sai], [p_volta, 1 - p_volta]])
        hmm.means_ = np.array([[xt[yt == 0].mean()], [xt[yt == 1].mean()]])
        hmm.covars_ = np.array([[max(xt[yt == 0].var(), self.var_min)],
                                [max(xt[yt == 1].var(), self.var_min)]])
        self.hmm = hmm
        self.estado_acordado = 1
        return self

    def prever(self, a, b):
        return (self.hmm.predict(self.x[a:b]) == self.estado_acordado).astype(int)


class Changepoints:

    def __init__(self, limiar, penalidades=(1, 3, 10, 30, 100), min_size=5):
        self.limiar, self.penalidades, self.min_size = limiar, penalidades, min_size
        self.pen = penalidades[len(penalidades) // 2]

    def _segmentar(self, a, b, pen):
        import ruptures as rpt
        bkps = rpt.KernelCPD(kernel='linear', min_size=self.min_size).fit(self.x[a:b]).predict(pen=pen)
        p = self.potencia[a:b]
        estados = np.zeros(b - a, dtype=int)
        ini = 0
        for fim in bkps:
            estados[ini:fim] = int(np.median(p[ini:fim]) > self.limiar)
            ini = fim
        return estados

    def ajustar_penalidade(self, potencia, idx_treino, avaliar):
        self.potencia = np.asarray(potencia, dtype=float)
        log_p = np.log(np.clip(self.potencia, 1e-3, None))
        # normalização com estatísticas do treino, para a penalidade valer igual em qualquer trecho
        self.x = ((log_p - log_p[idx_treino].mean()) / (log_p[idx_treino].std() + 1e-9)).reshape(-1, 1)
        melhor = (-1.0, self.pen)
        for pen in self.penalidades:
            f1 = avaliar(lambda a, b, pen=pen: self._segmentar(a, b, pen))
            print(f"  ruptures pen={pen}: F1-val={f1*100:.2f}%")
            melhor = max(melhor, (f1, pen))
        self.pen = melhor[1]
        return self

    def prever(self, a, b):
        return self._segmentar(a, b, self.pen)



# ============================== lstm.py ==============================
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def configurar_sementes(seed=42):
    import random
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def janelas(serie, alvos, seq_length):
    alvos = np.asarray(alvos, dtype=int)
    alvos = alvos[alvos >= seq_length - 1]
    vistas = np.lib.stride_tricks.sliding_window_view(serie, seq_length)
    return vistas[alvos - seq_length + 1][..., None].astype(np.float32), alvos


def build_lstm_model(seq_length=50):
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


class ModeloLSTM:
    def __init__(self, seq_length=50, epocas=100, batch_size=256, paciencia=5, passo_treino=1):
        self.seq, self.epocas, self.batch, self.paciencia = seq_length, epocas, batch_size, paciencia
        self.passo = max(1, passo_treino)
        self.limiar = 0.5

    def treinar(self, potencia, estados, idx_treino, idx_val):
        from sklearn.utils.class_weight import compute_class_weight
        from tensorflow.keras.callbacks import EarlyStopping
        p = np.asarray(potencia, dtype=float)
        lo, hi = p[idx_treino].min(), p[idx_treino].max()
        self.serie = np.clip((p - lo) / (hi - lo + 1e-9), -0.5, 1.5).astype(np.float32)

        X_tr, a_tr = janelas(self.serie, idx_treino[::self.passo], self.seq)
        X_va, a_va = janelas(self.serie, idx_val[::self.passo], self.seq)
        y_tr, y_va = estados[a_tr], estados[a_va]
        classes = np.unique(y_tr)
        pesos = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_tr)))
        print(f"LSTM — treino: {len(y_tr)} janelas | val: {len(y_va)} | pesos: {pesos}")

        self.model = build_lstm_model(self.seq)
        es = EarlyStopping(monitor='val_loss', patience=self.paciencia, restore_best_weights=True, verbose=1)
        self.history = self.model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=self.epocas,
                                      batch_size=self.batch, class_weight=pesos, callbacks=[es],
                                      shuffle=True, verbose=2)
        self.limiar = self._limiar_otimo(X_va, y_va)
        return self

    def _limiar_otimo(self, X_val, y_val):
        probs = self.model.predict(X_val, batch_size=4096, verbose=0).ravel()
        melhor_f1, melhor_t = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 81):
            pr = probs >= t
            tp = np.sum(pr & (y_val == 1)); fp = np.sum(pr & (y_val == 0)); fn = np.sum(~pr & (y_val == 1))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > melhor_f1:
                melhor_f1, melhor_t = f1, t
        print(f"✓ Limiar de probabilidade: {melhor_t:.2f} (F1-val por amostra={melhor_f1*100:.2f}%)")
        return melhor_t

    def prever(self, a, b):
        a_ef = max(a, self.seq - 1)
        X, _ = janelas(self.serie, np.arange(a_ef, b), self.seq)
        pred = np.zeros(b - a, dtype=int)
        if len(X):
            pred[a_ef - a:] = (self.model.predict(X, batch_size=4096, verbose=0).ravel() >= self.limiar)
        return pred



# ============================== main.py ==============================
ROTULOS_EMBUTIDOS = """\
ciclo_id,t0,t1,t2,revisado
1,67612.471,67619.082,67918.593,1
2,67918.593,67925.075,68224.648,1
3,68224.648,68231.284,68530.555,1
4,68530.555,68537.189,68836.744,1
5,68836.744,68843.361,69142.604,1
6,69142.604,69149.175,69448.556,1
7,69448.556,69455.202,69754.654,1
8,69754.654,69761.294,70060.473,1
9,70060.473,70067.117,70366.378,1
10,70366.378,70373.030,70672.465,1
11,70672.465,70679.107,70978.656,1
12,70978.656,70985.287,71284.727,1
13,71284.727,71291.381,71590.551,1
"""

MODELOS = ['limiar', 'heuristico', 'logreg', 'rf', 'hmm', 'ruptures', 'lstm']
NOMES = {'limiar': 'Limiar simples (Otsu)', 'heuristico': 'Heurístico (mediana+Otsu)',
         'logreg': 'Regressão Logística', 'rf': 'Random Forest', 'hmm': 'HMM 2 estados',
         'ruptures': 'ruptures (KernelCPD)', 'lstm': 'LSTM'}
MIN_DURACAO_S = 1.0   # pulsos mais curtos que isso são ruído (o período ativo dura ~6,5 s)
MARGEM_S = 10.0       # contexto extra em volta de cada trecho avaliado


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv-path', default=os.environ.get(
        'MONDESA_CSV_PATH', '/kaggle/input/datasets/biancatutihashi/experiment4-0/experiment4-0.CSV'))
    p.add_argument('--rotulos', default='', help='CSV ciclo_id,t0,t1,t2 (padrão: os rótulos embutidos)')
    p.add_argument('--output-dir', default=os.environ.get('MONDESA_OUTPUT_DIR', './resultados'))
    p.add_argument('--tolerancia', type=float, default=1.0,
                   help='janela de tolerância (s) para contar um evento como acerto')
    p.add_argument('--modelos', default=','.join(MODELOS), help=f'subconjunto de {",".join(MODELOS)}')
    p.add_argument('--epocas', type=int, default=100, help='máximo de épocas da LSTM (early stopping)')
    p.add_argument('--lstm-passo', type=int, default=1,
                   help='usa 1 a cada N janelas no treino da LSTM (acelera em CPU)')
    args, _ = p.parse_known_args(argv)
    args.modelos = [m.strip() for m in args.modelos.split(',') if m.strip()]
    desconhecidos = set(args.modelos) - set(MODELOS)
    if desconhecidos:
        p.error(f"modelos desconhecidos: {sorted(desconhecidos)}")
    return args


def obter_rotulos(args, tempo, potencia, ciclos_sinteticos):
    secao("FASE 2: RÓTULOS t0/t1/t2")
    if args.rotulos and os.path.exists(args.rotulos):
        ciclos, fonte = carregar_rotulos(args.rotulos), args.rotulos
    else:
        ciclos, fonte = carregar_rotulos(io.StringIO(ROTULOS_EMBUTIDOS)), 'embutidos (chunk 12 do coorientador)'
    print(f"✓ {len(ciclos)} ciclos rotulados: {fonte}\n")
    return ciclos, 'arquivo'


def dividir_ciclos(ciclos, frac_treino=0.6, frac_val=0.2):
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
    out = []
    for bloco in agrupar_ciclos_contiguos(ciclos):
        a, b = np.searchsorted(tempo, [bloco[0]['t0'] - margem_s, bloco[-1]['t2'] + margem_s])
        out.append((int(a), int(min(b, len(tempo)))))
    return out


def avaliar_preditor(prever, tempo, estados_reais, ciclos, tol_s):
    subidas, descidas, ciclos_pred, y_real, y_pred = [], [], [], [], []
    _, rotulado = ciclos_para_estados(tempo, ciclos)
    for a, b in trechos(tempo, ciclos):
        est = prever(a, b)
        s, d = estados_para_eventos(est, tempo[a:b], MIN_DURACAO_S)
        subidas += list(s); descidas += list(d)
        ciclos_pred += estados_para_ciclos(est, tempo[a:b], MIN_DURACAO_S)
        m = rotulado[a:b]
        y_real.append(estados_reais[a:b][m]); y_pred.append(est[m])
    r = avaliar_modelo(subidas, descidas, ciclos_pred, ciclos, tol_s)
    r.update(metricas_amostra(np.concatenate(y_real), np.concatenate(y_pred)))
    return r, ciclos_pred


def imprimir_resultado(nome, r, tol_s):
    print(f"{nome}: F1 eventos (±{tol_s}s)={r['f1']*100:.2f}% "
          f"[P={r['precisao']*100:.1f}% R={r['revocacao']*100:.1f}%] | "
          f"MAE t0={r['mae_t0_s']:.3f}s t1={r['mae_t1_s']:.3f}s t2={r['mae_t2_s']:.3f}s | "
          f"ciclos {r['ciclos_recuperados']}/{r['ciclos_reais']}")


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
    a, b = np.searchsorted(tempo, [ciclo['t0'] - 5, ciclo['t1'] + 5])
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(tempo[a:b], potencia[a:b], alpha=0.5, linewidth=0.8, label='Bruta')
    ax.plot(tempo[a:b], suave[a:b], linewidth=2, label='Suavizada (mediana)')
    _eixo(ax, 'Fase 1 — detecção heurística (zoom em um período ativo)', 'Tempo decorrido (s)', 'mW')
    plt.tight_layout(); return fig


def plotar_comparacao(tabela, tol_s):
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


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    tol = args.tolerancia

    # Fase 0
    ciclos_sint = None
    if not os.path.exists(args.csv_path):
        raise SystemExit(f"CSV não encontrado: {args.csv_path}")
    df, tempo, potencia = carregar_csv(args.csv_path)
    eda(df, tempo, potencia)
    figs = {"00_eda": plotar_eda(tempo, potencia)}

    # Fase 2 (vem antes da modelagem: define treino/validação/teste)
    ciclos, origem = obter_rotulos(args, tempo, potencia, ciclos_sint)
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
    est_heur, suave = estados_heuristico(potencia, tempo, limiar, min_duracao_s=MIN_DURACAO_S)
    preditores = {}
    if 'limiar' in args.modelos:
        bruto = (potencia > limiar).astype(int)
        preditores['limiar'] = lambda a, b: bruto[a:b]
    if 'heuristico' in args.modelos:
        preditores['heuristico'] = lambda a, b: est_heur[a:b]
        figs["04_heuristico"] = plotar_heuristico(tempo, potencia, suave, c_te[0])

    def f1_val(prever):
        return avaliar_preditor(prever, tempo, estados, c_va, tol)[0]['f1']

    # Fase 3
    if {'logreg', 'rf'} & set(args.modelos):
        secao("FASE 3: CLASSIFICAÇÃO PONTO A PONTO (features de janela)")
        feats = calcular_features(potencia)
        for tipo in ('logreg', 'rf'):
            if tipo in args.modelos:
                t_ini = time.time()
                preditores[tipo] = ClassificadorJanela(tipo).treinar(feats, estados, idx_tr).prever
                print(f"✓ {NOMES[tipo]} treinado em {time.time() - t_ini:.1f}s")

    # Fase 4
    if {'hmm', 'ruptures'} & set(args.modelos):
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
        secao("FASE 6: LSTM")
        configurar_sementes(42)
        lstm = ModeloLSTM(epocas=args.epocas, passo_treino=args.lstm_passo).treinar(potencia, estados, idx_tr, idx_va)
        preditores['lstm'] = lstm.prever
        figs["01_historico_lstm"] = plotar_historico(lstm.history)

    # Fase 5
    secao(f"FASE 5: AVALIAÇÃO POR EVENTO NO TESTE ({len(c_te)} ciclos, tolerância ±{tol}s)")
    linhas = []
    for nome, prever in preditores.items():
        r, _ = avaliar_preditor(prever, tempo, estados, c_te, tol)
        imprimir_resultado(NOMES[nome], r, tol)
        linhas.append({'modelo': NOMES[nome], **r})
    tabela = pd.DataFrame(linhas).sort_values(['f1', 'mae_t0_s'], ascending=[False, True])
    cols = ['modelo', 'f1', 'precisao', 'revocacao', 'f1_subida', 'f1_descida', 'mae_t0_s', 'mae_t1_s',
            'mae_t2_s', 'ciclos_recuperados', 'ciclos_reais', 'acuracia_amostra', 'f1_amostra']
    tabela = tabela[cols]
    tabela['rotulos'] = origem
    caminho_tab = os.path.join(args.output_dir, 'comparacao_modelos.csv')
    tabela.to_csv(caminho_tab, index=False, float_format='%.4f')

    secao("COMPARAÇÃO FINAL (ordenada por F1 de eventos)")
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(tabela.drop(columns=['rotulos']).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAcurácia por amostra aparece só como referência: ~2% das amostras são ACORDADO,\n"
          f"então ela fica perto de 100% mesmo para modelos que erram os instantes.")

    figs["02_previsoes_teste"] = plotar_previsoes_teste(tempo, potencia, c_te, preditores)
    figs["03_comparacao_modelos"] = plotar_comparacao(tabela, tol)
    # conferência: as linhas t0/t1/t2 precisam cair em cima dos degraus de potência
    figs["05_rotulos"] = plotar_ciclos_rotulados(tempo, potencia, ciclos, f"Rótulos ({origem}) — conferência")
    salvar_graficos(figs, os.path.join(args.output_dir, "pdf"))
    print(f"\n✓ Pipeline concluído. Tabela: {caminho_tab} | gráficos: {os.path.join(args.output_dir, 'pdf')}")
    return tabela


if __name__ == "__main__":
    main()
