"""Fases 3 e 4 — modelos clássicos e específicos de série temporal.

Todos seguem a mesma interface: `prever(a, b)` devolve o estado (0/1) das
amostras a..b-1. A partir daí, a extração de t0/t1/t2 e a avaliação são
as mesmas para todo modelo (ver ciclos.py e avaliacao.py).

Fase 3 (classificação ponto a ponto): Regressão Logística e Random Forest
sobre features de janela causal (só amostras até t, como seria num nodo
sensor em tempo real). As features são estatísticas simples de janelas
rolantes; o `tsfresh` sugerido no roadmap faria o mesmo de forma
automática, mas fica lento demais para uma janela por amostra em ~1,3
milhão de amostras.

Fase 4: HMM gaussiano de 2 estados (`hmmlearn`) e detecção de pontos de
mudança (`ruptures`).
"""
import numpy as np
import pandas as pd

JANELAS = (5, 20, 60)  # em amostras (~0,3 s, ~1,1 s, ~3,4 s com passo de 57 ms)


def calcular_features(potencia):
    """Features de janela causal para cada amostra (só usam o passado e o presente)."""
    p = pd.Series(np.asarray(potencia, dtype=float))
    f = {'p': p, 'log_p': np.log(p.clip(lower=1e-3)), 'diff1': p.diff(1), 'diff5': p.diff(5)}
    for w in JANELAS:
        r = p.rolling(w, min_periods=1)
        f[f'media_{w}'] = r.mean(); f[f'mediana_{w}'] = r.median()
        f[f'std_{w}'] = r.std(); f[f'min_{w}'] = r.min(); f[f'max_{w}'] = r.max()
    return pd.DataFrame(f).fillna(0.0).to_numpy(dtype=np.float32)


def _amostrar_treino(idx, y, max_amostras, seed=0):
    """Mantém todos os positivos (raros) e uma amostra dos negativos."""
    if len(idx) <= max_amostras:
        return idx
    rng = np.random.default_rng(seed)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = rng.choice(neg, size=max(0, max_amostras - len(pos)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


class ClassificadorJanela:
    """Fase 3: classificador scikit-learn sobre `calcular_features`."""

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
    """Fase 4: HMM gaussiano com 2 estados ocultos sobre log(potência).

    HMM supervisado: transições, médias e variâncias são estimadas direto
    dos rótulos de treino e a decodificação é por Viterbi. Não usamos EM:

    - com as transições livres, o EM aprende a explicar cada pico isolado de
      ruído (~100 mW durante o sono) como um "acordado" curtíssimo;
    - a potência do sensor é quantizada (dormindo só aparece 45,5 / 51,9 /
      58,4 mW) e o EM colapsa a variância do sono num único valor; aí 45 ou
      58 mW passam a parecer "acordado". Por isso há também um piso de variância.

    A entrada passa pelo mesmo filtro de mediana da heurística.
    """

    def __init__(self, var_min=0.01):
        self.var_min = var_min  # em log(mW)²: desvio de ~10% da potência

    def treinar(self, potencia, tempo, estados, idx_treino):
        from hmmlearn.hmm import GaussianHMM
        from heuristico import suavizar
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
    """Fase 4: `ruptures` (KernelCPD, núcleo linear = custo L2, implementação em C).

    Os pontos de mudança dividem o sinal em segmentos; cada segmento vira
    ACORDADO se a mediana da potência nele passar do limiar de Otsu. A
    penalidade (quantos pontos de mudança aceitar) é escolhida na validação.
    """

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
        """`avaliar(prever)` devolve o F1 de eventos na validação para um preditor."""
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


class ClassificadorCiclos:
    """Classifica cada ciclo como regular ou falho a partir das suas medidas (T_on, T, P_on, P_off, P_max).

    Treinado nos ciclos VERDADEIROS do treino e aplicado aos ciclos que cada
    modelo detectou no teste: um erro de detecção (t0/t1 no lugar errado)
    distorce as medidas e aparece como erro de classificação, como deve ser.
    Se o treino só tem uma classe (ex.: exp4, todos regulares), prevê essa classe.
    """

    def __init__(self, seed=42):
        self.seed = seed

    @staticmethod
    def _x(medidas):
        from ciclos import MEDIDAS
        return np.nan_to_num(np.array([[m[k] for k in MEDIDAS] for m in medidas], dtype=float))

    def treinar(self, medidas, estados):
        estados = np.asarray(estados)
        self.classes = np.unique(estados)
        if len(self.classes) == 1:
            self.modelo = None
            return self
        from sklearn.ensemble import RandomForestClassifier
        self.modelo = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced',
                                             random_state=self.seed).fit(self._x(medidas), estados)
        return self

    def prever(self, medidas):
        if not len(medidas):
            return np.array([], dtype=object)
        if self.modelo is None:
            return np.full(len(medidas), self.classes[0], dtype=object)
        return self.modelo.predict(self._x(medidas))
