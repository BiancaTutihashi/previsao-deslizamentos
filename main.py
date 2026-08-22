"""
Projeto Mondesa - Pipeline Completo de Machine Learning
Classificação de Estados (DORMINDO vs ACORDADO) via LSTM
COM DETECÇÃO DE CICLOS (t0, t1, t2)

ARQUITETURA:
- Fase 0: EDA e exploração de dados
- Fase 1: Baseline heurístico (Savitzky-Golay + threshold)
- Fase 2-3: Preparação e divisão estratificada (SEM SHUFFLE - crítico para séries temporais)
- Fase 4: Treinamento LSTM com Early Stopping
- Fase 5: Detecção de transições e ciclos
- Fase 6: Comparação e avaliação (LSTM vs Heurístico)
- Fase 7: Visualização e exportação (PDF + PNG + ZIP)

NOTAS IMPORTANTES:
1. Ordem temporal PRESERVADA: não embaralhamos teste/validação (shuffle=False implícito)
2. Índices mapeados: conseguimos recuperar tempo/potência originais do conjunto de teste
3. Dual-path evaluation: comparamos 2 métodos (ML + heurístico) para validar resultados
4. Early Stopping monitora val_loss, não accuracy (mais estável para classificação binária)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

# Core libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Deep Learning stack
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Feature scaling e model selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Metrics e evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Signal processing
from scipy.signal import savgol_filter
import plotly.express as px

#%%

# ==========================================
# PARTE 0: FUNÇÕES AUXILIARES (EDA E ANÁLISE)
# ==========================================

def eda_inspect_csv(path, time_col_guess=None, power_col_guess=None, show_plot=True):
    """Exploração inicial dos dados (Fase 0 - Exploratory Data Analysis)."""
    df = pd.read_csv(path)
    
    print("Shape:", df.shape)
    print("\nTipos de dados:")
    print(df.dtypes)
    print("\nNulos por coluna:\n", df.isnull().sum())
    print("\nPrimeiras linhas:")
    print(df.head())

    # Auto-detecção de colunas: time
    if time_col_guess is None:
        candidates = [c for c in df.columns 
                     if any(x in c.lower() for x in ['tempo', 'time', 'delta', 'timestamp'])]
        time_col = candidates[0] if candidates else df.columns[0]
    else:
        time_col = time_col_guess

    # Auto-detecção de colunas: power
    if power_col_guess is None:
        candidates = [c for c in df.columns 
                     if any(x in c.lower() for x in ['consumed', 'power', 'mw', 'potencia'])]
        power_col = candidates[0] if candidates else df.columns[1]
    else:
        power_col = power_col_guess

    print(f"\nUsando coluna de tempo: '{time_col}'; potência: '{power_col}'")

    # IMPORTANTE: Checar regularidade temporal
    # Séries temporais irregulares degradam performance do LSTM
    if np.issubdtype(df[time_col].dtype, np.number):
        diffs = df[time_col].diff().dropna().values
        print("\n⚠️ IMPORTANTE - Passo temporal (regularity check):")
        print(f"  min={np.min(diffs):.4f}, mean={np.mean(diffs):.4f}, "
              f"median={np.median(diffs):.4f}, max={np.max(diffs):.4f}")
        
        cv = (np.std(diffs) / np.mean(diffs)) * 100
        if cv > 10:
            print(f"  ⚠️ ATENÇÃO: Coeficiente de variação = {cv:.1f}% (série irregular!)")
    else:
        print("⚠️ Coluna de tempo não numérica — conversão necessária")

    if show_plot:
        try:
            fig = px.line(df, x=time_col, y=power_col, 
                         title='Série de Potência (interativo - use zoom/pan)')
            fig.show()
        except Exception as e:
            print(f"Erro ao gerar plot Plotly: {e}")

    return df, time_col, power_col


def detectar_ciclos_heuristico(potencia, tempo, threshold, min_duracao_s=0.5, 
                               window_len=51, polyorder=3):
    """Detecção de ciclos usando método heurístico (Fase 1 - Baseline)."""
    n = len(potencia)
    if n <= 3:
        raise ValueError("Sinal muito curto para suavização (n≤3)")
    
    # Validação de parâmetros Savitzky-Golay
    if window_len >= n:
        window_len = n - 1 if ((n - 1) % 2 == 1) else n - 2
    if window_len % 2 == 0:
        window_len -= 1
        if window_len < 3:
            window_len = 3

    try:
        potencia_suave = savgol_filter(potencia, window_length=window_len, polyorder=polyorder)
    except Exception as e:
        print(f"  Savitzky-Golay falhou ({e}), usando média móvel como fallback")
        window = max(1, int(window_len // 3))
        potencia_suave = pd.Series(potencia).rolling(
            window=window, min_periods=1, center=True).mean().values

    # Detecção de bordas
    acima = potencia_suave > threshold
    dif = np.diff(acima.astype(int))
    
    on_idx = np.where(dif == 1)[0] + 1
    off_idx = np.where(dif == -1)[0] + 1

    # Tratar bordas do sinal
    if acima[0]:
        on_idx = np.concatenate(([0], on_idx))
    if acima[-1]:
        off_idx = np.concatenate((off_idx, [len(acima)]))

    # Filtrar por duração mínima
    ciclos = []
    for on, off in zip(on_idx, off_idx):
        dur = float(tempo[off - 1] - tempo[on])
        if dur >= min_duracao_s:
            ciclos.append({
                't0': float(tempo[on]),
                't1': float(tempo[off - 1]),
                't2': None,
                'duracao_acordado': dur
            })
    
    print(f"Detecção heurística: {len(ciclos)} eventos detectados (threshold={threshold:.1f} mW)")
    return ciclos, potencia_suave


def avaliar_transicoes(pred_times, true_times, tolerance_s=1.0):
    """Avaliação de transições com tolerância (Fase 5)."""
    pred = sorted(list(pred_times))
    true = sorted(list(true_times))
    used_true = set()
    tp = 0
    fp = 0
    errors = []

    for p in pred:
        candidates = [(i, abs(p - t)) for i, t in enumerate(true) if i not in used_true]
        
        if not candidates:
            fp += 1
            continue
        
        best_idx, best_diff = min(candidates, key=lambda x: x[1])
        
        if best_diff <= tolerance_s:
            tp += 1
            used_true.add(best_idx)
            errors.append(best_diff)
        else:
            fp += 1

    fn = len(true) - len(used_true)
    mae = float(np.mean(errors)) if errors else None
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'mae_s': mae
    }


#%%

# ==========================================
# PARTE 1: PREPARAÇÃO DE DADOS - SEQUÊNCIAS
# ==========================================

def preparar_sequencias(dados, targets, tempo=None, seq_length=50):
    """Converte série temporal em sequências (sliding window)."""
    X_seq = []
    y_seq = []
    label_indices = []

    for i in range(len(dados) - seq_length):
        X_seq.append(dados[i:i + seq_length])
        y_seq.append(targets[i + seq_length])
        label_indices.append(i + seq_length)

    X_seq = np.array(X_seq).reshape(-1, seq_length, 1)
    y_seq = np.array(y_seq)
    label_indices = np.array(label_indices, dtype=int)

    if tempo is not None:
        tempo_labels = np.array(tempo)[label_indices]
        return X_seq, y_seq, label_indices, tempo_labels
    
    return X_seq, y_seq, label_indices


#%%

# ==========================================
# PARTE 2: DIVISÃO DE DADOS - SEM SHUFFLE (CRÍTICO!)
# ==========================================

def dividir_dataset_estratificado_sem_shuffle(X_seq, y_seq, label_indices, seed=42):
    """Divide dataset em treino/validação/teste PRESERVANDO ORDEM TEMPORAL."""
    print("Dividindo dataset preservando ordem temporal (60/20/20)...\n")
    
    n = len(y_seq)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    
    # CRUCIAL: não embaralhar! Série temporal tem dependência temporal.
    # Se embaralharmos, o modelo "vê o futuro" durante validação → validação enviesada
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    
    X_treino = X_seq[train_idx]
    y_treino = y_seq[train_idx]
    X_val = X_seq[val_idx]
    y_val = y_seq[val_idx]
    X_teste = X_seq[test_idx]
    y_teste = y_seq[test_idx]
    
    # Mapear índices originais (CRÍTICO para recuperar tempo/potência depois)
    original_idx_treino = label_indices[train_idx]
    original_idx_val = label_indices[val_idx]
    original_idx_teste = label_indices[test_idx]
    
    # Diagnóstico: verificar desbalanceamento de classes
    print(f"Conjunto de treinamento: {X_treino.shape[0]} amostras ({X_treino.shape[0]/len(X_seq)*100:.1f}%)")
    print(f"  - DORMINDO (0): {np.sum(y_treino == 0)} ({np.sum(y_treino == 0)/len(y_treino)*100:.1f}%)")
    print(f"  - ACORDADO (1): {np.sum(y_treino == 1)} ({np.sum(y_treino == 1)/len(y_treino)*100:.1f}%)\n")
    
    print(f"Conjunto de validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X_seq)*100:.1f}%)")
    print(f"  - DORMINDO (0): {np.sum(y_val == 0)} ({np.sum(y_val == 0)/len(y_val)*100:.1f}%)")
    print(f"  - ACORDADO (1): {np.sum(y_val == 1)} ({np.sum(y_val == 1)/len(y_val)*100:.1f}%)\n")
    
    print(f"Conjunto de teste: {X_teste.shape[0]} amostras ({X_teste.shape[0]/len(X_seq)*100:.1f}%)")
    print(f"  - DORMINDO (0): {np.sum(y_teste == 0)} ({np.sum(y_teste == 0)/len(y_teste)*100:.1f}%)")
    print(f"  - ACORDADO (1): {np.sum(y_teste == 1)} ({np.sum(y_teste == 1)/len(y_teste)*100:.1f}%)\n")
    
    return (X_treino, X_val, X_teste,
            y_treino, y_val, y_teste,
            original_idx_treino, original_idx_val, original_idx_teste)


#%%

# ==========================================
# PARTE 3: ARQUITETURA LSTM
# ==========================================

def build_lstm_model(seq_length=50):
    """Constrói modelo LSTM para classificação binária."""
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(seq_length, 1), name='lstm_1'),
        Dropout(0.2, name='dropout_1'),
        LSTM(units=32, return_sequences=False, name='lstm_2'),
        Dropout(0.2, name='dropout_2'),
        Dense(units=16, activation='relu', name='dense_1'),
        Dropout(0.2, name='dropout_3'),
        Dense(units=1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


#%%

# ==========================================
# PARTE 4: TREINAMENTO COM EARLY STOPPING
# ==========================================

def treinar_modelo(X_treino, y_treino, X_val, y_val, epochs=100, batch_size=32, patience=5):
    """Treina modelo LSTM com validação e Early Stopping."""
    model = build_lstm_model(seq_length=X_treino.shape[1])

    # Early Stopping: estratégia anti-overfitting
    # Se val_loss não melhorar por N épocas, para e restaura melhor checkpoint
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    print(f"Iniciando treinamento do modelo LSTM (com Early Stopping)...")
    print(f"  - Paciência: {patience} épocas")
    print(f"  - Máximo de épocas: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Treino: {X_treino.shape[0]} amostras")
    print(f"  - Validação: {X_val.shape[0]} amostras\n")

    history = model.fit(
        X_treino, y_treino,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


#%%

# ==========================================
# PARTE 5: DETECÇÃO DE TRANSIÇÕES
# ==========================================

def detectar_transicoes(previsoes, tempo, min_duracao=1.0):
    """Encontra instantes onde o estado muda (transições)."""
    previsoes = np.array(previsoes).astype(int)
    
    transicoes = np.diff(previsoes) != 0
    indices_transicoes = np.where(transicoes)[0] + 1

    if len(indices_transicoes) == 0:
        print("⚠️ Nenhuma transição detectada!")
        return np.array([]), np.array([]), np.array([])

    tempos_transicoes = np.array(tempo)[indices_transicoes]
    estados_antes = previsoes[indices_transicoes - 1]

    # Filtrar transições muito próximas (ruído)
    transicoes_validas = []
    for i, t in enumerate(tempos_transicoes):
        if i == 0:
            transicoes_validas.append(i)
        else:
            duracao_desde_ultima = t - tempos_transicoes[transicoes_validas[-1]]
            if duracao_desde_ultima >= min_duracao:
                transicoes_validas.append(i)

    indices_transicoes = indices_transicoes[transicoes_validas]
    tempos_transicoes = tempos_transicoes[transicoes_validas]
    estados_transicoes = previsoes[indices_transicoes - 1].astype(int)

    print(f"✓ {len(tempos_transicoes)} transições detectadas\n")

    return indices_transicoes, tempos_transicoes, estados_transicoes


#%%

# ==========================================
# PARTE 6: AGRUPAMENTO EM CICLOS COMPLETOS
# ==========================================

def extrair_ciclos(tempos_transicoes, estados_transicoes):
    """Agrupa transições em ciclos completos (t0, t1, t2)."""
    ciclos = []
    i = 0

    while i + 2 < len(tempos_transicoes):
        # Padrão: 0→1 (acordou), 1→0 (dormiu), 0→1 (acordou de novo)
        if (estados_transicoes[i] == 0 and
            estados_transicoes[i + 1] == 1 and
            estados_transicoes[i + 2] == 0):

            ciclo = {
                'ciclo_num': len(ciclos) + 1,
                't0': float(tempos_transicoes[i]),
                't1': float(tempos_transicoes[i + 1]),
                't2': float(tempos_transicoes[i + 2]),
                'duracao_acordado': float(tempos_transicoes[i + 1] - tempos_transicoes[i]),
                'duracao_dormindo': float(tempos_transicoes[i + 2] - tempos_transicoes[i + 1]),
                'duracao_ciclo': float(tempos_transicoes[i + 2] - tempos_transicoes[i]),
            }
            ciclos.append(ciclo)
            i += 2
        else:
            i += 1

    print(f"✓ {len(ciclos)} ciclos completos extraídos\n")

    return ciclos


#%%

# ==========================================
# PARTE 7: ANÁLISE ESTATÍSTICA DE CICLOS
# ==========================================

def analisar_ciclos(ciclos):
    """Calcula estatísticas dos ciclos detectados."""
    if len(ciclos) == 0:
        print("⚠️ Nenhum ciclo para analisar!")
        return None

    duracao_acordado = np.array([c['duracao_acordado'] for c in ciclos])
    duracao_dormindo = np.array([c['duracao_dormindo'] for c in ciclos])
    duracao_ciclo = np.array([c['duracao_ciclo'] for c in ciclos])

    stats = {
        'num_ciclos': len(ciclos),
        'duracao_acordado_media': np.mean(duracao_acordado),
        'duracao_acordado_std': np.std(duracao_acordado),
        'duracao_dormindo_media': np.mean(duracao_dormindo),
        'duracao_dormindo_std': np.std(duracao_dormindo),
        'duracao_ciclo_media': np.mean(duracao_ciclo),
        'duracao_ciclo_std': np.std(duracao_ciclo),
        'uniformidade_acordado': (np.std(duracao_acordado) / np.mean(duracao_acordado)) * 100,
        'uniformidade_dormindo': (np.std(duracao_dormindo) / np.mean(duracao_dormindo)) * 100,
        'uniformidade_ciclo': (np.std(duracao_ciclo) / np.mean(duracao_ciclo)) * 100,
    }

    return stats


#%%

# ==========================================
# PARTE 8: MÉTRICAS DE CLASSIFICAÇÃO
# ==========================================

def gerar_metricas_classificacao(y_real, y_pred):
    """Gera relatório completo de classificação."""
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    report = classification_report(
        y_real, y_pred,
        target_names=['DORMINDO (0)', 'ACORDADO (1)'],
        digits=4
    )
    print(report)

    accuracy = accuracy_score(y_real, y_pred)
    print(f"Accuracy Global: {accuracy * 100:.2f}%\n")

    print("=" * 80)
    print("MATRIZ DE CONFUSÃO")
    print("=" * 80)
    cm = confusion_matrix(y_real, y_pred)
    print(f"\n{cm}\n")
    print("Interpretação:")
    print(f"  Verdadeiro Negativo (TN):  {cm[0, 0]:>5d}  (DORMINDO correto)")
    print(f"  Falso Positivo (FP):       {cm[0, 1]:>5d}  (DORMINDO → ACORDADO incorreto)")
    print(f"  Falso Negativo (FN):       {cm[1, 0]:>5d}  (ACORDADO → DORMINDO incorreto)")
    print(f"  Verdadeiro Positivo (TP):  {cm[1, 1]:>5d}  (ACORDADO correto)\n")
    print("=" * 80 + "\n")

    return cm


#%%

# ==========================================
# PARTE 9: VISUALIZAÇÃO - MATRIZ DE CONFUSÃO
# ==========================================

def plotar_matriz_confusao(y_real, y_pred):
    """Desenha a matriz de confusão em heatmap."""
    cm = confusion_matrix(y_real, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['DORMINDO', 'ACORDADO'],
                yticklabels=['DORMINDO', 'ACORDADO'],
                cbar_kws={'label': 'Contagem'},
                annot_kws={'size': 14, 'weight': 'bold'})

    ax.set_title('Matriz de Confusão - Modelo LSTM', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rótulo Real', fontsize=12)
    ax.set_xlabel('Rótulo Predito', fontsize=12)

    plt.tight_layout()
    return fig


#%%

# ==========================================
# PARTE 10: VISUALIZAÇÃO - CICLOS LSTM
# ==========================================

def plotar_ciclos_com_transicoes(tempo_teste, potencia_teste, previsoes_finais, ciclos, recorte=10000):
    """Plota os ciclos detectados pelo modelo LSTM."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Gráfico 1: Potência Bruta
    ax1.plot(tempo_teste[:recorte], potencia_teste[:recorte],
             color="#1f77b4", linewidth=1.2, label="Potência Medida")

    for ciclo in ciclos:
        if ciclo['t2'] <= tempo_teste[recorte - 1]:
            ax1.axvspan(ciclo['t0'], ciclo['t1'], alpha=0.15, color="green")
            ax1.axvspan(ciclo['t1'], ciclo['t2'], alpha=0.15, color="purple")

    ax1.set_title("Potência Consumida (Dados de Teste)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Potência (mW)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc='upper right')

    # Gráfico 2: Previsões LSTM
    ax2.plot(tempo_teste[:recorte], previsoes_finais[:recorte],
             color="#ff7f0e", linewidth=0.5, alpha=0.7, label="Previsão LSTM")

    for ciclo in ciclos:
        if ciclo['t2'] <= tempo_teste[recorte - 1]:
            ax2.axvspan(ciclo['t0'], ciclo['t1'], alpha=0.15, color="green")
            ax2.axvspan(ciclo['t1'], ciclo['t2'], alpha=0.15, color="purple")

    plotados = {'t0': False, 't1': False, 't2': False}

    for ciclo in ciclos:
        if ciclo['t2'] <= tempo_teste[recorte - 1]:
            if not plotados['t0']:
                ax2.plot(ciclo['t0'], 1, 'o', color='green', markersize=8,
                        label=f"t0 (DORMINDO→ACORDADO)", zorder=5)
                plotados['t0'] = True
            else:
                ax2.plot(ciclo['t0'], 1, 'o', color='green', markersize=8, zorder=5)

            if not plotados['t1']:
                ax2.plot(ciclo['t1'], 0, 's', color='red', markersize=8,
                        label=f"t1 (ACORDADO→DORMINDO)", zorder=5)
                plotados['t1'] = True
            else:
                ax2.plot(ciclo['t1'], 0, 's', color='red', markersize=8, zorder=5)

            if not plotados['t2']:
                ax2.plot(ciclo['t2'], 1, '^', color='blue', markersize=8,
                        label=f"t2 (DORMINDO→ACORDADO)", zorder=5)
                plotados['t2'] = True
            else:
                ax2.plot(ciclo['t2'], 1, '^', color='blue', markersize=8, zorder=5)

    ax2.set_title("Previsão LSTM com Ciclos Detectados (0=Dormindo, 1=Acordado)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tempo Decorrido (s)")
    ax2.set_ylabel("Estado Previsto")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['DORMINDO', 'ACORDADO'])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


#%%

# ==========================================
# PARTE 10B: VISUALIZAÇÃO - CICLOS HEURÍSTICOS
# ==========================================

def plotar_ciclos_heuristico(tempo_teste, potencia_teste, potencia_suave, ciclos_heur, recorte=10000):
    """Plota ciclos detectados pelo método heurístico."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(tempo_teste[:recorte], potencia_teste[:recorte],
             color="#1f77b4", linewidth=0.8, alpha=0.6, label="Potência Bruta (medida)")
    ax1.plot(tempo_teste[:recorte], potencia_suave[:recorte],
             color="#ff7f0e", linewidth=2, label="Potência Suavizada (Savitzky-Golay)")

    ax1.set_title("Detecção Heurística - Potência Bruta vs Suavizada", 
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Potência (mW)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc='upper right')

    # Reconstruir estado binário
    estado_heur = np.zeros(len(tempo_teste))
    for ciclo in ciclos_heur:
        mask = (tempo_teste >= ciclo['t0']) & (tempo_teste <= ciclo['t1'])
        estado_heur[mask] = 1

    ax2.plot(tempo_teste[:recorte], estado_heur[:recorte],
             color="#2ca02c", linewidth=1, drawstyle='steps-post', 
             label="Estado Heurístico (0=Dormindo, 1=Acordado)")
    ax2.fill_between(tempo_teste[:recorte], 0, estado_heur[:recorte], 
                     alpha=0.3, color="#2ca02c")

    plotados = {'t0': False, 't1': False}
    for ciclo in ciclos_heur:
        if ciclo['t1'] <= tempo_teste[recorte - 1]:
            if not plotados['t0']:
                ax2.plot(ciclo['t0'], 1, 'o', color='green', markersize=8,
                        label="t0 (DORMINDO→ACORDADO)", zorder=5)
                plotados['t0'] = True
            else:
                ax2.plot(ciclo['t0'], 1, 'o', color='green', markersize=8, zorder=5)

            if not plotados['t1']:
                ax2.plot(ciclo['t1'], 0, 's', color='red', markersize=8,
                        label="t1 (ACORDADO→DORMINDO)", zorder=5)
                plotados['t1'] = True
            else:
                ax2.plot(ciclo['t1'], 0, 's', color='red', markersize=8, zorder=5)

    ax2.set_title("Estados Detectados (Heurístico)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tempo Decorrido (s)")
    ax2.set_ylabel("Estado")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['DORMINDO', 'ACORDADO'])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


#%%

# ==========================================
# PARTE 11: VISUALIZAÇÃO - HISTÓRICO DE TREINAMENTO
# ==========================================

def plotar_historico_treinamento(history):
    """Plota curvas de loss e accuracy durante o treinamento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Loss durante o Treinamento', fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss (erro)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_title('Accuracy durante o Treinamento', fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


#%%

# ==========================================
# PARTE 12: TABELA DE CICLOS
# ==========================================

def exibir_tabela_ciclos(ciclos, stats):
    """Exibe tabela formatada com ciclos e estatísticas."""
    print("=" * 80)
    print("CICLOS DETECTADOS - RESUMO DETALHADO")
    print("=" * 80)
    print(f"\n{'Ciclo':<8} {'t0 (s)':<12} {'t1 (s)':<12} {'t2 (s)':<12} "
          f"{'Acordado (s)':<15} {'Dormindo (s)':<15}")
    print("-" * 80)

    for ciclo in ciclos:
        print(f"{ciclo['ciclo_num']:<8} {ciclo['t0']:<12.2f} {ciclo['t1']:<12.2f} "
              f"{ciclo['t2']:<12.2f} {ciclo['duracao_acordado']:<15.2f} "
              f"{ciclo['duracao_dormindo']:<15.2f}")

    print("\n" + "=" * 80)
    print("ESTATÍSTICAS DE UNIFORMIDADE")
    print("=" * 80)
    print(f"Total de ciclos: {stats['num_ciclos']}\n")

    print("PERÍODO ACORDADO:")
    print(f"  Média: {stats['duracao_acordado_media']:.2f}s")
    print(f"  Desvio Padrão: {stats['duracao_acordado_std']:.2f}s")
    print(f"  Coeficiente de Variação: {stats['uniformidade_acordado']:.2f}%\n")

    print("PERÍODO DORMINDO:")
    print(f"  Média: {stats['duracao_dormindo_media']:.2f}s")
    print(f"  Desvio Padrão: {stats['duracao_dormindo_std']:.2f}s")
    print(f"  Coeficiente de Variação: {stats['uniformidade_dormindo']:.2f}%\n")

    print("CICLO COMPLETO:")
    print(f"  Média: {stats['duracao_ciclo_media']:.2f}s")
    print(f"  Desvio Padrão: {stats['duracao_ciclo_std']:.2f}s")
    print(f"  Coeficiente de Variação: {stats['uniformidade_ciclo']:.2f}%\n")

    print("QUALIDADE DOS CICLOS:")
    if stats['uniformidade_ciclo'] < 10:
        print("  ✓ Excelente! Ciclos muito uniformes (< 10% variação)")
    elif stats['uniformidade_ciclo'] < 20:
        print("  ✓ Bom! Ciclos razoavelmente uniformes (< 20% variação)")
    else:
        print("  ⚠️ Atenção! Ciclos com alta variação (> 20%)")

    print("=" * 80 + "\n")


#%%

# ==========================================
# PARTE 13: EXPORTAÇÃO - PDF E PNG
# ==========================================

def salvar_graficos_pdf(figs_dict, output_dir="/kaggle/working/resultados_pdf"):
    """Salva figuras matplotlib em PDF."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("SALVANDO GRÁFICOS EM PDF")
    print("=" * 80 + "\n")

    for nome, fig in figs_dict.items():
        pdf_path = os.path.join(output_dir, f"{timestamp}_{nome}.pdf")
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {pdf_path}")

    print(f"\n✓ Todos os gráficos foram salvos em '{output_dir}'\n")
    print("=" * 80 + "\n")


def salvar_graficos_png(figs_dict, output_dir="/kaggle/working/resultados_png"):
    """Salva figuras matplotlib em PNG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("SALVANDO GRÁFICOS EM PNG")
    print("=" * 80 + "\n")

    for nome, fig in figs_dict.items():
        png_path = os.path.join(output_dir, f"{timestamp}_{nome}.png")
        fig.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
        print(f"✓ Salvo: {png_path}")

    print(f"\n✓ Todos os gráficos foram salvos em '{output_dir}'\n")
    print("=" * 80 + "\n")


def criar_zip_resultados(output_dir="/kaggle/working/resultados_pdf"):
    """Agrupa todos os PDFs em um ZIP para download único."""
    import zipfile
    import glob
    
    if not os.path.exists(output_dir):
        print(f"Diretório {output_dir} não encontrado!")
        return
    
    zip_path = "/kaggle/working/resultados_mondesa.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        pdfs = glob.glob(f"{output_dir}/*.pdf")
        for pdf in pdfs:
            zipf.write(pdf, arcname=os.path.basename(pdf))
    
    print(f"\n✓ ZIP criado com sucesso: {zip_path}")
    print(f"✓ Total de PDFs no ZIP: {len(pdfs)}")
    print(f"✓ Tamanho do ZIP: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB\n")


#%%

# ==========================================
# PARTE 14: ORQUESTRAÇÃO - MAIN
# ==========================================

def main():
    """Pipeline completo de ML para detecção de ciclos em séries de potência."""
    
    # === PASSO 1: CARREGAR DADOS ===
    csv_path = "/kaggle/input/datasets/biancatutihashi/experiment4-0/experiment4-0.CSV"
    print("A carregar o ficheiro CSV...")
    df = pd.read_csv(csv_path)
    print("Ficheiro carregado com sucesso!\n")

    # === PASSO 2: PRÉ-PROCESSAMENTO ===
    # Construir tempo contínuo
    if "Delta-T_[ms]" in df.columns:
        df["Tempo_Decorrido_[s]"] = df["Delta-T_[ms]"].cumsum() / 1000.0
    else:
        time_cols = [c for c in df.columns 
                    if any(x in c.lower() for x in ['tempo', 'time', 'timestamp'])]
        if time_cols:
            df["Tempo_Decorrido_[s]"] = df[time_cols[0]].astype(float)
        else:
            df["Tempo_Decorrido_[s]"] = np.arange(len(df))

    # Criar rótulos automáticos (gabarito por limiar)
    LIMIAR_MW = 100
    if "Consumed_[mW]" in df.columns:
        df["Estado_Real"] = (df["Consumed_[mW]"] > LIMIAR_MW).astype(int)
    else:
        power_cols = [c for c in df.columns 
                     if any(x in c.lower() for x in ['consumed', 'power', 'mw', 'potencia'])]
        if power_cols:
            df["Estado_Real"] = (df[power_cols[0]] > LIMIAR_MW).astype(int)
            df.rename(columns={power_cols[0]: "Consumed_[mW]"}, inplace=True)
        else:
            raise RuntimeError("Coluna de potência não encontrada!")

    # Extrair arrays
    potencia = df["Consumed_[mW]"].values.astype(float)
    estados = df["Estado_Real"].values.astype(int)
    tempo_total = df["Tempo_Decorrido_[s]"].values.astype(float)

    # Normalizar potência
    scaler = MinMaxScaler()
    potencia_normalizada = scaler.fit_transform(potencia.reshape(-1, 1)).flatten()

    # === PASSO 3: PREPARAR SEQUÊNCIAS ===
    print("Preparando sequências para LSTM...")
    SEQ_LENGTH = 50
    X_seq, y_seq, label_indices, tempo_labels = preparar_sequencias(
        potencia_normalizada, estados, tempo=tempo_total, seq_length=SEQ_LENGTH)
    print(f"✓ Sequências preparadas: {X_seq.shape}\n")

    # === PASSO 4: DIVIDIR DADOS (SEM SHUFFLE - CRÍTICO!) ===
    (X_treino, X_val, X_teste,
     y_treino, y_val, y_teste,
     idx_treino, idx_val, idx_teste) = dividir_dataset_estratificado_sem_shuffle(X_seq, y_seq, label_indices)

    # Recuperar tempo/potência do conjunto de teste
    tempo_teste = tempo_total[idx_teste]
    potencia_teste = potencia[idx_teste]
    
    print("Conjunto de teste (primeiros/últimos tempos):")
    print(f"  Início: {tempo_teste[0]:.2f}s, Fim: {tempo_teste[-1]:.2f}s\n")

    # === PASSO 5: TREINAR LSTM ===
    model, history = treinar_modelo(
        X_treino, y_treino, X_val, y_val,
        epochs=100,
        batch_size=32,
        patience=3
    )

    # === PASSO 6: AVALIAR MODELO ===
    print("\nAvaliando o modelo LSTM nos dados de teste...")
    test_loss, test_accuracy = model.evaluate(X_teste, y_teste, verbose=0)
    print(f"Precisão final (Accuracy): {test_accuracy * 100:.2f}%\n")

    # === PASSO 7: FAZER PREVISÕES ===
    previsoes_prob = model.predict(X_teste, verbose=0)
    previsoes_finais = (previsoes_prob > 0.5).astype(int).flatten()

    # === PASSO 8: MÉTRICAS DE CLASSIFICAÇÃO ===
    print("\n")
    gerar_metricas_classificacao(y_teste, previsoes_finais)

    # === PASSO 9: BASELINE HEURÍSTICO ===
    print("=" * 80)
    print("EXECUTANDO DETECTOR HEURÍSTICO (BASELINE)")
    print("=" * 80 + "\n")
    
    # Limiar automático
    limiar_auto = np.mean(potencia_teste[potencia_teste > 50])
    if np.isnan(limiar_auto) or limiar_auto < 50:
        limiar_auto = 100
    
    print(f"Limiar automático detectado: {limiar_auto:.1f} mW\n")
    
    ciclos_heur, potencia_suave = detectar_ciclos_heuristico(
        potencia_teste, tempo_teste,
        threshold=limiar_auto * 0.8,
        min_duracao_s=0.5,
        window_len=51,
        polyorder=3
    )

    # === PASSO 10: DETECÇÃO DE TRANSIÇÕES (LSTM) ===
    print("=" * 80)
    print("DETECÇÃO DE TRANSIÇÕES DO MODELO LSTM")
    print("=" * 80 + "\n")
    
    indices_transicoes, tempos_transicoes, estados_transicoes = detectar_transicoes(
        previsoes_finais, tempo_teste, min_duracao=1.0
    )

    # === PASSO 11: EXTRAIR CICLOS (LSTM) ===
    ciclos_lstm = extrair_ciclos(tempos_transicoes, estados_transicoes)

    # === PASSO 12: ANÁLISE ESTATÍSTICA (LSTM) ===
    print("\n📊 ANÁLISE LSTM:")
    if len(ciclos_lstm) > 0:
        stats_lstm = analisar_ciclos(ciclos_lstm)
        exibir_tabela_ciclos(ciclos_lstm, stats_lstm)
    else:
        print("⚠️ LSTM: Nenhum ciclo completo detectado.\n")
        stats_lstm = None

    # === PASSO 13: ANÁLISE ESTATÍSTICA (HEURÍSTICO) ===
    print("\n📊 ANÁLISE HEURÍSTICA:")
    if len(ciclos_heur) > 0:
        ciclos_heur_completos = []
        for i in range(0, len(ciclos_heur) - 1):
            if i + 1 < len(ciclos_heur):
                ciclo = {
                    'ciclo_num': i + 1,
                    't0': ciclos_heur[i]['t0'],
                    't1': ciclos_heur[i]['t1'],
                    't2': ciclos_heur[i + 1]['t0'],
                    'duracao_acordado': ciclos_heur[i]['duracao_acordado'],
                    'duracao_dormindo': ciclos_heur[i+1]['t0'] - ciclos_heur[i]['t1'],
                    'duracao_ciclo': ciclos_heur[i+1]['t0'] - ciclos_heur[i]['t0'],
                }
                ciclos_heur_completos.append(ciclo)
        
        if ciclos_heur_completos:
            stats_heur = analisar_ciclos(ciclos_heur_completos)
            exibir_tabela_ciclos(ciclos_heur_completos, stats_heur)
        else:
            print("⚠️ Heurístico: Não conseguiu formar ciclos completos.\n")
            stats_heur = None
    else:
        print("⚠️ Heurístico: Nenhum evento detectado.\n")
        stats_heur = None

    # === PASSO 14: GERAR FIGURAS ===
    print("\nGerando gráficos...\n")
    fig_historico = plotar_historico_treinamento(history)
    fig_confusao = plotar_matriz_confusao(y_teste, previsoes_finais)

    RECORTE = min(10000, len(tempo_teste))
    fig_lstm = plotar_ciclos_com_transicoes(tempo_teste, potencia_teste, previsoes_finais,
                                            ciclos_lstm, recorte=RECORTE)
    fig_heur = plotar_ciclos_heuristico(tempo_teste, potencia_teste, potencia_suave,
                                        ciclos_heur, recorte=RECORTE)

    # === PASSO 15: SALVAR E EXPORTAR ===
    figs_para_salvar = {
        "01_historico_treinamento": fig_historico,
        "02_matriz_confusao": fig_confusao,
        "03_ciclos_lstm": fig_lstm,
        "04_ciclos_heuristico": fig_heur
    }

    salvar_graficos_pdf(figs_para_salvar, output_dir="/kaggle/working/resultados_pdf")
    salvar_graficos_png(figs_para_salvar, output_dir="/kaggle/working/resultados_png")
    
    try:
        from IPython.display import display
        print("\n📊 Exibindo gráficos no notebook Kaggle...\n")
        for nome, fig in figs_para_salvar.items():
            print(f"\n--- {nome.upper()} ---")
            display(fig)
    except ImportError:
        print("⚠️ IPython não disponível, gráficos salvos em arquivo.\n")
    
    plt.show()

    criar_zip_resultados("/kaggle/working/resultados_pdf")
    
    plt.close('all')
    
    print("\n✓ Pipeline completo executado com sucesso!\n")


if __name__ == "__main__":
    main()