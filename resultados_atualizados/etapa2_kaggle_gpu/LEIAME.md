# Rodada no Kaggle (GPU), 10/11/2026

Mesmo código da comparação em `../etapa2_rastreamento/` (arquivo único gerado no commit 8a124fd),
rodado no Kaggle com GPU para treinar a LSTM com **todas** as janelas de treino
(a rodada em CPU usou 1 a cada 11–22 janelas).

- Datasets: exp8, exp12-0, exp15-0 e exp11. O exp4 não rodou (dados ou gabarito ausentes nos inputs).
- Métodos: limiar simples, heurística, Regressão Logística, Random Forest e LSTM. O HMM e o
  ruptures foram pulados porque `hmmlearn` e `ruptures` não estavam instalados no notebook.

F1 de eventos da LSTM: CPU (amostrada) → Kaggle (completa)

| exp8 | exp12-0 | exp15-0 | exp11 |
|---|---|---|---|
| 51% → 66% | 79% → 81% | 59% → 59% | 22% → 22% |

Os demais métodos deram os mesmos números da rodada em CPU, exceto o Random Forest no exp8
(62,5% em CPU, 57% no Kaggle), provavelmente por versões diferentes do scikit-learn.
