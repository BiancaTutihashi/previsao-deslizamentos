# previsao-deslizamentos

## Contexto

Este repositório é a primeira etapa concreta de um projeto de pesquisa (ICMC-USP)
sobre redes de sensores sem fio (WSN) para monitoramento de deslizamentos de terra.

Para que uma WSN de monitoramento seja viável em campo, os nodos sensores
precisam gerenciar bateria de forma inteligente (*duty-cycling*). Antes de
otimizar isso, é preciso **identificar automaticamente, a partir do consumo
de energia, os ciclos de trabalho de cada nodo**.

## Objetivo

Treinar um modelo que identifique, numa série temporal de potência, três
instantes por ciclo:

- **t0**: início do ciclo (início do período ativo);
- **t1**: fim do período ativo, início do período dormindo;
- **t2**: fim do período dormindo, que é o t0 do ciclo seguinte.

O plano é começar por um dataset "fácil", o `experiment4-0`, que tem ciclos
uniformes (~6,6 s ativo a cada ~306 s) e degraus nítidos (≈55 mW dormindo,
>150 mW ativo). Depois o trabalho avança para dados mais difíceis.

Referência teórica: o problema é próximo de **NILM (Non-Intrusive Load
Monitoring)**, a área que estuda detecção de transições de estado em sinais de
potência.

## Estrutura (uma fase do roadmap por módulo)

| Fase | Arquivo | O que faz |
|---|---|---|
| 0 – Dados | `dados.py` | carrega o CSV, resumo (colunas, tipos, nulos, regularidade do passo), gráfico da série inteira + recorte, HTML interativo (plotly) |
| 1 – Heurística | `heuristico.py` | filtro de mediana + limiar de Otsu + duração mínima → lista de (t0, t1, t2) |
| 2 – Rótulos | `rotulagem.py`, `rotulos/` | formato `ciclo_id,t0,t1,t2`, ferramenta para propor/revisar/conferir rótulos |
| 3 – ML ponto a ponto | `modelos.py` | Regressão Logística e Random Forest com features de janela |
| 4 – Série temporal | `modelos.py` | HMM de 2 estados (`hmmlearn`) e changepoints (`ruptures`) |
| 5 – Avaliação | `avaliacao.py` | MAE em segundos para t0/t1/t2, precisão/revocação/F1 de eventos com tolerância |
| 6 – Deep learning | `lstm.py` | LSTM (TensorFlow) |
| — | `ciclos.py` | conversões estado ponto a ponto ↔ transições ↔ ciclos |
| — | `main.py` | roda todas as fases e gera a tabela comparativa |

## Como rodar

```bash
pip install -r requirements.txt
python main.py --csv-path experiment4-0.CSV --output-dir ./resultados
```

Opções úteis:

- `--rotulos rotulos/rotulos_ciclos.csv`: arquivo de rótulos (esse é o padrão);
- `--tolerancia 1.0`: janela de tolerância, em segundos, para contar um evento como acerto;
- `--modelos limiar,heuristico,logreg,rf,hmm,ruptures,lstm`: escolhe quais modelos rodar;
- `--epocas 100` e `--lstm-passo 1`: para a LSTM. Um `--lstm-passo` maior usa menos janelas no treino e acelera em CPU.

Sem o CSV, o pipeline roda numa **série sintética** parecida com a real, em que
os t0/t1/t2 verdadeiros são conhecidos. Serve para testar o código, não como
resultado.

Saídas em `--output-dir`:

- `comparacao_modelos.csv`: tabela final, uma linha por modelo;
- `pdf/*_00_eda.pdf`: série inteira e recorte de alguns ciclos (Fase 0);
- `pdf/*_01_historico_lstm.pdf`: curvas de treino da LSTM;
- `pdf/*_02_previsoes_teste.pdf`: zoom num período ativo do teste, com o estado previsto por cada modelo;
- `pdf/*_03_comparacao_modelos.pdf`: F1 de eventos e MAE de t0/t1/t2 por modelo;
- `pdf/*_04_heuristico.pdf`: sinal bruto × suavizado;
- `pdf/*_05_rotulos.pdf`: os ciclos rotulados (até 13), para conferir se t0/t1/t2 caem nos degraus;
- `html/eda_interativo.html`: série com zoom.

Testes: `pytest -v`.

## Rótulos (Fase 2)

Os rótulos ficam em `rotulos/rotulos_ciclos.csv`, uma linha por ciclo, com os
tempos em segundos no eixo "tempo decorrido" (soma acumulada de `Delta-T_[ms]`).
O mesmo arquivo serve para treinar **e** avaliar todos os modelos. A divisão é
cronológica por ciclo: 60% treino, 20% validação e 20% teste.

**Estado atual:** o arquivo tem os 13 ciclos do *chunk 12* (67612–71590 s),
**transcritos da figura enviada pelo coorientador**. A transcrição é
consistente (ativo = 6,62 ± 0,05 s, dormindo = 299,4 ± 0,15 s, t2 = t0 do ciclo
seguinte), mas foi feita a partir de uma imagem. O ideal é trocar pelo arquivo
original. Com 13 ciclos sobram só 2 ou 3 para teste, então vale rotular mais
trechos.

Para rotular mais trechos:

```bash
# 1. propõe ciclos num trecho (pela heurística) + gráficos para revisão
python rotulagem.py propor --csv-path experiment4-0.CSV --inicio 20000 --fim 24000
# 2. abra rotulos/revisao.html (o hover mostra o tempo exato de cada ponto),
#    corrija rotulos/rotulos_propostos.csv e marque revisado=1
# 3. junte as linhas em rotulos/rotulos_ciclos.csv e confira:
python rotulagem.py visualizar --csv-path experiment4-0.CSV --rotulos rotulos/rotulos_ciclos.csv
```

Ciclos com `revisado=0` são ignorados. A proposta vem da heurística da Fase 1
e só serve para economizar digitação. Se for aceita sem conferir, o gabarito
vira a própria heurística e a comparação volta a ser circular.

Se o arquivo de rótulos não existir, o pipeline cai em **rótulos provisórios**:
um limiar fixo de 100 mW sobre a própria potência, que era a regra antiga deste
repositório. Nesse caso ele avisa em destaque, porque as métricas passam a medir
só o quanto os modelos reproduzem essa regra.

## Avaliação (Fase 5)

A acurácia por amostra **não** é a métrica principal: só ~2% das amostras são
"acordado", e os instantes de transição são raríssimos. Um modelo que erra t0
por vários segundos ainda passa de 99% de acurácia. As métricas principais são:

- **F1 / precisão / revocação de eventos**: uma subida (t0/t2) ou descida (t1)
  prevista conta como acerto se ficar a no máximo `--tolerancia` segundos de um
  evento real, com casamento um-para-um. Eventos fora dos trechos rotulados são
  ignorados;
- **MAE (s) de t0, t1 e t2**: erro médio absoluto, em segundos, de cada instante;
- **ciclos recuperados**: ciclos reais com t0, t1 e t2 todos dentro da tolerância.

Todos os modelos passam pela mesma rotina: cada um produz o estado ponto a
ponto, e a extração de t0/t1/t2 (com descarte de pulsos < 1 s) e as métricas
são as mesmas para todos.

## Modelos

- **Limiar simples (Otsu)**: `potência > limiar`. O limiar é estimado por Otsu
  só na potência do treino, sem ver rótulo nenhum. É a regra mais simples
  possível; um modelo de ML só se justifica se superá-la.
- **Heurístico (Fase 1)**: filtro de mediana (~0,3 s) + limiar de Otsu +
  duração mínima. Substitui o Savitzky-Golay anterior, que "afundava" o sinal
  para ~25 mW antes de cada subida (ringing em degrau). A mediana remove os
  picos isolados de ruído sem deformar a borda do degrau.
- **Regressão Logística / Random Forest (Fase 3)**: features de janelas
  causais de 5, 20 e 60 amostras (média, mediana, desvio, mín., máx.,
  diferenças). O `tsfresh` faria isso automaticamente, mas fica lento demais
  para uma janela por amostra em ~1,3 milhão de amostras.
- **HMM de 2 estados (Fase 4)**: gaussiano sobre log(potência suavizada). As
  probabilidades de transição vêm dos rótulos de treino. Se o EM também
  ajustasse as transições, ele aprenderia a explicar cada pico de ruído como um
  "acordado" curtíssimo.
- **ruptures (Fase 4)**: `KernelCPD` (custo L2, implementação em C). Cada
  segmento vira "acordado" se a mediana da potência nele passar do limiar de
  Otsu. A penalidade é escolhida na validação pelo F1 de eventos.
- **LSTM (Fase 6)**: 2 camadas LSTM sobre janelas de 50 amostras terminando no
  ponto classificado, com pesos de classe e limiar de probabilidade escolhidos
  na validação.

Todas as janelas são **causais** (só usam amostras até o instante t), como
seria num nodo sensor em tempo real. Isso deixa o caminho aberto para a etapa
de Edge AI.

## Histórico

- A versão anterior usava rótulos tirados do limiar de 100 mW e um baseline com
  essa mesma regra, que acertava 100% por construção. Isso foi corrigido em
  `88e3e83` (limiar de Otsu, baseline com `x[t-1]`). Agora o problema de fundo
  também está resolvido: os rótulos vêm de fora do sinal (Fase 2).
- A opção `--horizonte` (prever N amostras à frente) foi removida. O objetivo
  do projeto é identificar t0/t1/t2 na série, não prever o estado futuro. Ela
  só existia para contornar a circularidade dos rótulos antigos.
- `resultados_atualizados/` guarda os gráficos da versão antiga (classificação
  por amostra) para referência.

## Próximos passos

1. Substituir a transcrição pelo arquivo original de rótulos do coorientador e
   rotular mais trechos (algumas dezenas de ciclos).
2. Rodar o pipeline completo no `experiment4-0` (Kaggle) e registrar a tabela.
3. Ferramentas de apoio quando houver mais modelos e datasets: `optuna`
   (hiperparâmetros) e `mlflow`/`wandb` (registro de experimentos).
4. Datasets mais difíceis; se os modelos simples deixarem de bastar, testar
   1D-CNN/TCN (por exemplo com `tsai`).

## Arquivo único para o Kaggle

Para rodar no Kaggle sem subir a pasta inteira, gere um `.py` autossuficiente,
com os rótulos de `rotulos/rotulos_ciclos.csv` embutidos:

```bash
python gerar_arquivo_unico.py            # completo -> mondesa_kaggle.py (~1000 linhas)
python gerar_arquivo_unico.py --enxuto   # enxuto   -> mondesa_kaggle_enxuto.py (~850 linhas)
```

A versão enxuta tira o que não roda no Kaggle (série sintética, rótulos
provisórios, gráficos HTML) e as docstrings. Os resultados são os mesmos.

No notebook (com Internet ligada):

```python
!pip install -q hmmlearn ruptures
!python mondesa_kaggle_enxuto.py --csv-path /kaggle/input/.../experiment4-0.CSV --output-dir /kaggle/working/resultados
```

Os arquivos gerados não são versionados. Para mudar o código, edite os módulos e gere de novo.
