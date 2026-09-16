"""Projeto Mondesa — LSTM DORMINDO/ACORDADO, com baseline heurístico e trivial."""
import os, random, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from scipy.signal import savgol_filter


def configurar_sementes(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.random.set_seed(seed); tf.keras.utils.set_random_seed(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv-path', default=os.environ.get(
        'MONDESA_CSV_PATH', '/kaggle/input/datasets/biancatutihashi/experiment4-0/experiment4-0.CSV'))
    p.add_argument('--output-dir', default=os.environ.get('MONDESA_OUTPUT_DIR', './resultados'))
    args, _ = p.parse_known_args()
    return args


def secao(titulo):
    print("=" * 80); print(titulo); print("=" * 80)


def _calcular_cv_seguro(vetor):
    if len(vetor) == 0:
        return 0.0
    media, desvio = float(np.mean(vetor)), float(np.std(vetor))
    if media == 0 or np.isnan(media) or np.isnan(desvio):
        return 0.0
    return (desvio / media) * 100.0


def eda_inspect_csv(path):
    df = pd.read_csv(path)
    secao("FASE 0: EDA")
    print(f"Shape: {df.shape} | Nulos: {df.isnull().sum().sum()}")
    time_col = next((c for c in df.columns if any(x in c.lower() for x in ['tempo', 'time', 'delta'])), df.columns[0])
    if np.issubdtype(df[time_col].dtype, np.number):
        diffs = df[time_col].diff().dropna().values
        cv = _calcular_cv_seguro(diffs)
        status = "⚠️ irregular" if cv > 10 else "✓ regular"
        print(f"Passo temporal ({time_col}): média={np.mean(diffs):.4f}s, CV={cv:.2f}% ({status})")
    print()
    return df


def detectar_ciclos_heuristico(potencia, tempo, threshold=100.0, min_duracao_s=0.5, window_len=51, polyorder=3):
    n = len(potencia)
    if n <= 3:
        raise ValueError("Sinal muito curto")
    if window_len >= n:
        window_len = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_len % 2 == 0:
        window_len = max(3, window_len - 1)
    try:
        potencia_suave = savgol_filter(potencia, window_length=window_len, polyorder=polyorder)
    except Exception:
        w = max(1, window_len // 3)
        potencia_suave = pd.Series(potencia).rolling(window=w, min_periods=1, center=True).mean().values

    acima = potencia_suave > threshold
    dif = np.diff(acima.astype(int))
    on_idx = np.where(dif == 1)[0] + 1
    off_idx = np.where(dif == -1)[0] + 1
    if acima[0]:
        on_idx = np.concatenate(([0], on_idx))
    if acima[-1]:
        off_idx = np.concatenate((off_idx, [len(acima)]))

    ciclos = [{'t0': float(tempo[on]), 't1': float(tempo[off - 1]), 't2': None,
               'duracao_acordado': float(tempo[off - 1] - tempo[on])}
              for on, off in zip(on_idx, off_idx) if tempo[off - 1] - tempo[on] >= min_duracao_s]
    print(f"Heurística: {len(ciclos)} eventos (threshold={threshold:.1f} mW)")
    return ciclos, potencia_suave


def avaliar_transicoes(pred_times, true_times, tolerance_s=1.0):
    pred, true = sorted(pred_times), sorted(true_times)
    used, tp, fp, errors = set(), 0, 0, []
    for p in pred:
        cand = [(i, abs(p - t)) for i, t in enumerate(true) if i not in used]
        if not cand:
            fp += 1; continue
        i, d = min(cand, key=lambda x: x[1])
        if d <= tolerance_s:
            tp += 1; used.add(i); errors.append(d)
        else:
            fp += 1
    fn = len(true) - len(used)
    mae = float(np.mean(errors)) if errors else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    secao(f"AVALIAÇÃO DE TRANSIÇÕES (tol={tolerance_s}s)")
    print(f"TP={tp} FP={fp} FN={fn} | Precisão={precision*100:.2f}% Revocação={recall*100:.2f}% "
          f"F1={f1*100:.2f}% MAE={mae:.4f}s\n")
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'mae_s': mae}


def preparar_sequencias(dados, targets, tempo=None, seq_length=50):
    X, y, idx = [], [], []
    for i in range(len(dados) - seq_length):
        X.append(dados[i:i + seq_length]); y.append(targets[i + seq_length]); idx.append(i + seq_length)
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)
    idx = np.array(idx, dtype=int)
    if tempo is not None:
        return X, y, idx, np.array(tempo)[idx]
    return X, y, idx


def dividir_dados_brutos_cronologico(potencia, estados, tempo):
    """Corte cronológico simples (60/20/20) — sem estratificação, sem shuffle."""
    n = len(potencia)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    return np.arange(0, n_tr), np.arange(n_tr, n_tr + n_va), np.arange(n_tr + n_va, n)


def build_lstm_model(seq_length=50):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def treinar_modelo(X_tr, y_tr, X_val, y_val, epochs=100, batch_size=32, patience=5, class_weights=None):
    model = build_lstm_model(seq_length=X_tr.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    print(f"Treino: {X_tr.shape[0]} amostras | Val: {X_val.shape[0]} | Pesos: {class_weights}")
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                         batch_size=batch_size, class_weight=class_weights, callbacks=[es],
                         shuffle=True, verbose=2)
    return model, history


def encontrar_limiar_otimo(model, X_val, y_val):
    probs = model.predict(X_val, verbose=0).flatten()
    melhor_f1, melhor_t = -1.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1)); fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > melhor_f1:
            melhor_f1, melhor_t = f1, t
    print(f"✓ Limiar ótimo: {melhor_t:.2f} (F1-val={melhor_f1*100:.2f}%)")
    return melhor_t


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


def detectar_transicoes(previsoes, tempo, min_duracao=1.0):
    suave = aplicar_debouncing_estados(previsoes, tempo, min_duracao)
    idx = np.where(np.diff(suave) != 0)[0] + 1
    if len(idx) == 0:
        print("⚠️ Nenhuma transição detectada.")
        return np.array([]), np.array([]), np.array([])
    print(f"✓ {len(idx)} transições detectadas.")
    return idx, np.array(tempo)[idx], suave[idx]


def extrair_ciclos(tempos, estados):
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
    print(f"✓ {len(ciclos)} ciclos extraídos.\n")
    return ciclos


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


def obter_metricas_dict(y_real, y_pred):
    r = classification_report(y_real, y_pred, target_names=['DORMINDO (0)', 'ACORDADO (1)'],
                               output_dict=True, zero_division=0)
    return {'accuracy': accuracy_score(y_real, y_pred), 'precision': r['ACORDADO (1)']['precision'],
            'recall': r['ACORDADO (1)']['recall'], 'f1': r['ACORDADO (1)']['f1-score']}


def gerar_metricas_classificacao(y_real, y_pred):
    secao("RELATÓRIO DE CLASSIFICAÇÃO (LSTM)")
    print(classification_report(y_real, y_pred, target_names=['DORMINDO (0)', 'ACORDADO (1)'], digits=4))
    print(f"Acurácia: {accuracy_score(y_real, y_pred)*100:.2f}%")
    print("Matriz de confusão:\n", confusion_matrix(y_real, y_pred), "\n")


def avaliar_baseline_trivial(potencia, y_real, threshold=100.0):
    """Baseline x[t] > threshold, sem modelo — o comparador que a revisão pedia."""
    m = obter_metricas_dict(y_real, (potencia > threshold).astype(int))
    secao(f"BASELINE TRIVIAL: x[t] > {threshold:.1f} mW")
    print(f"Acurácia={m['accuracy']*100:.2f}% Precisão={m['precision']*100:.2f}% "
          f"Revocação={m['recall']*100:.2f}% F1={m['f1']*100:.2f}%\n")
    return m


def comparar_lstm_vs_trivial(lstm, trivial):
    secao("LSTM vs. BASELINE TRIVIAL (nível de amostra)")
    for chave, nome in [('accuracy', 'Acurácia'), ('precision', 'Precisão'),
                         ('recall', 'Revocação'), ('f1', 'F1-Score')]:
        d = lstm[chave]*100 - trivial[chave]*100
        veredito = "✓ LSTM" if d > 0.01 else ("≈ empate" if abs(d) <= 0.01 else "✗ trivial")
        print(f"{nome:<12} LSTM={lstm[chave]*100:6.2f}  Trivial={trivial[chave]*100:6.2f}  Δ={d:+.2f}pp ({veredito})")
    print()


def _eixo(ax, titulo, xlabel, ylabel, legenda=True):
    ax.set_title(titulo, fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if legenda:
        ax.legend()


def plotar_historico(history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history['loss'], label='Treino'); a1.plot(history.history['val_loss'], label='Val')
    _eixo(a1, 'Loss', 'Época', 'Loss')
    a2.plot(history.history['accuracy'], label='Treino'); a2.plot(history.history['val_accuracy'], label='Val')
    _eixo(a2, 'Acurácia', 'Época', 'Acurácia')
    plt.tight_layout(); return fig


def plotar_matriz_confusao(y_real, y_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y_real, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['DORMINDO', 'ACORDADO'], yticklabels=['DORMINDO', 'ACORDADO'])
    _eixo(ax, 'Matriz de Confusão - LSTM', 'Predito', 'Real', legenda=False)
    plt.tight_layout(); return fig


def plotar_ciclos_lstm(tempo, potencia, previsoes, ciclos, recorte=10000):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    lim = min(recorte, len(tempo))
    a1.plot(tempo[:lim], potencia[:lim], linewidth=1.2, label="Potência"); _eixo(a1, "Potência (Teste)", "", "mW")
    a2.plot(tempo[:lim], previsoes[:lim], linewidth=1, label="Previsão LSTM")
    marcado = {'t0': False, 't1': False, 't2': False}
    for c in ciclos:
        if c['t2'] <= tempo[lim - 1]:
            for k, (marker, cor, y) in zip(['t0', 't1', 't2'], [('o', 'green', 1), ('s', 'red', 0), ('^', 'blue', 1)]):
                a2.plot(c[k], y, marker, color=cor, markersize=7, label=k if not marcado[k] else None)
                marcado[k] = True
    a2.set_yticks([0, 1]); a2.set_yticklabels(['DORMINDO', 'ACORDADO'])
    _eixo(a2, "Transições t0/t1/t2", "Tempo (s)", "Estado")
    plt.tight_layout(); return fig


def plotar_ciclos_heuristico(tempo, potencia, potencia_suave, recorte=10000):
    fig, ax = plt.subplots(figsize=(14, 5))
    lim = min(recorte, len(tempo))
    ax.plot(tempo[:lim], potencia[:lim], alpha=0.4, label="Bruta")
    ax.plot(tempo[:lim], potencia_suave[:lim], linewidth=2, label="Suavizada")
    _eixo(ax, "Detecção Heurística", "Tempo (s)", "mW")
    plt.tight_layout(); return fig


def exibir_tabela_ciclos(ciclos, stats, limite=20):
    if not ciclos or not stats:
        print("⚠️ Sem ciclos suficientes.\n"); return
    secao(f"RESUMO DOS CICLOS ({stats['num_ciclos']} encontrados)")
    for c in ciclos[:limite]:
        print(f"#{c['ciclo_num']:<4} t0={c['t0']:.2f} t1={c['t1']:.2f} t2={c['t2']:.2f} "
              f"acordado={c['duracao_acordado']:.2f}s dormindo={c['duracao_dormindo']:.2f}s")
    if len(ciclos) > limite:
        print(f"... (+{len(ciclos) - limite} ciclos omitidos)")
    print(f"\nMédia acordado={stats['duracao_acordado_media']:.2f}s (CV={stats['uniformidade_acordado']:.2f}%) | "
          f"dormindo={stats['duracao_dormindo_media']:.2f}s (CV={stats['uniformidade_dormindo']:.2f}%)\n")


def salvar_graficos(figs, output_dir, formato="png", dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for nome, fig in figs.items():
        fig.savefig(os.path.join(output_dir, f"{ts}_{nome}.{formato}"), format=formato, dpi=dpi, bbox_inches='tight')


def main():
    configurar_sementes(42)
    args = parse_args()
    LIMIAR_MW = 100.0

    if not os.path.exists(args.csv_path):
        print(f"⚠️ {args.csv_path} não encontrado — usando série sintética.")
        t = np.linspace(0, 1000, 10000)
        p = np.where((t % 100) > 80, 250 + np.random.normal(0, 5, len(t)), 10 + np.random.normal(0, 2, len(t)))
        df = pd.DataFrame({"Delta-T_[ms]": np.ones(len(t)) * 100, "Consumed_[mW]": p})
    else:
        df = eda_inspect_csv(args.csv_path)

    df["Tempo_Decorrido_[s]"] = df["Delta-T_[ms]"].cumsum() / 1000.0 if "Delta-T_[ms]" in df.columns else np.arange(len(df))
    df["Estado_Real"] = (df["Consumed_[mW]"] > LIMIAR_MW).astype(int)

    potencia = df["Consumed_[mW]"].values.astype(float)
    estados = df["Estado_Real"].values.astype(int)
    tempo = df["Tempo_Decorrido_[s]"].values.astype(float)

    idx_tr, idx_va, idx_te = dividir_dados_brutos_cronologico(potencia, estados, tempo)

    scaler = MinMaxScaler()
    pot_tr = scaler.fit_transform(potencia[idx_tr].reshape(-1, 1)).flatten()
    pot_va = scaler.transform(potencia[idx_va].reshape(-1, 1)).flatten()
    pot_te = scaler.transform(potencia[idx_te].reshape(-1, 1)).flatten()

    SEQ = 50
    X_tr, y_tr, _, _ = preparar_sequencias(pot_tr, estados[idx_tr], tempo[idx_tr], SEQ)
    X_va, y_va, _, _ = preparar_sequencias(pot_va, estados[idx_va], tempo[idx_va], SEQ)
    X_te, y_te, _, tempo_te = preparar_sequencias(pot_te, estados[idx_te], tempo[idx_te], SEQ)
    potencia_te = potencia[idx_te][SEQ:]
    estados_te_real = estados[idx_te][SEQ:]

    classes = np.unique(y_tr)
    pesos = compute_class_weight('balanced', classes=classes, y=y_tr)
    dict_pesos = dict(zip(classes, pesos))

    model, history = treinar_modelo(X_tr, y_tr, X_va, y_va, class_weights=dict_pesos)
    limiar_otimo = encontrar_limiar_otimo(model, X_va, y_va)
    previsoes = (model.predict(X_te, verbose=0).flatten() >= limiar_otimo).astype(int)

    gerar_metricas_classificacao(y_te, previsoes)
    metricas_lstm = obter_metricas_dict(y_te, previsoes)
    metricas_trivial = avaliar_baseline_trivial(potencia_te, y_te, threshold=LIMIAR_MW)
    comparar_lstm_vs_trivial(metricas_lstm, metricas_trivial)

    ciclos_heur, potencia_suave = detectar_ciclos_heuristico(potencia_te, tempo_te, threshold=LIMIAR_MW)
    _, tempos_lstm, estados_lstm = detectar_transicoes(previsoes, tempo_te)
    ciclos_lstm = extrair_ciclos(tempos_lstm, estados_lstm)
    _, tempos_reais, _ = detectar_transicoes(estados_te_real, tempo_te)

    print("\n[DUAL-PATH] LSTM:")
    avaliar_transicoes(tempos_lstm, tempos_reais, tolerance_s=2.0)

    # t0 (liga) + t1 (desliga) — usar só t0 travava o recall em 50%
    tempos_heur = sorted([c['t0'] for c in ciclos_heur] + [c['t1'] for c in ciclos_heur])
    print("[DUAL-PATH] Heurístico:")
    avaliar_transicoes(tempos_heur, tempos_reais, tolerance_s=2.0)

    if ciclos_lstm:
        exibir_tabela_ciclos(ciclos_lstm, analisar_ciclos(ciclos_lstm))

    figs = {
        "01_historico_treinamento": plotar_historico(history),
        "02_matriz_confusao": plotar_matriz_confusao(y_te, previsoes),
        "03_ciclos_lstm": plotar_ciclos_lstm(tempo_te, potencia_te, previsoes, ciclos_lstm),
        "04_ciclos_heuristico": plotar_ciclos_heuristico(tempo_te, potencia_te, potencia_suave),
    }
    salvar_graficos(figs, os.path.join(args.output_dir, "png"))
    print("\n✓ Pipeline concluído.")


if __name__ == "__main__":
    main()