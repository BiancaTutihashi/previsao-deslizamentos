"""Fase 6 — LSTM (deep learning), avaliada com as mesmas métricas de evento dos outros modelos.

A janela de entrada termina na própria amostra classificada (janela
causal, como num nodo sensor em tempo real). Isso só é honesto porque os
rótulos agora vêm de fora do sinal (Fase 2); com rótulos tirados de um
limiar sobre a potência, a tarefa seria trivial.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np


def configurar_sementes(seed=42):
    import random
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def janelas(serie, alvos, seq_length):
    """X[k] = serie[alvo_k - seq_length + 1 : alvo_k + 1] (alvos com histórico insuficiente são descartados)."""
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
