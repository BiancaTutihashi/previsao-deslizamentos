# previsao-deslizamentos

## Contexto

Este repositório é a primeira etapa concreta de um projeto de pesquisa (ICMC-USP)
sobre redes de sensores sem fio (WSN) para monitoramento de deslizamentos de terra.

Para que uma WSN de monitoramento seja viável em campo, os nodos sensores
precisam gerenciar bateria de forma inteligente (*duty-cycling*). Antes de
otimizar isso, é preciso conseguir **identificar automaticamente, a partir do
consumo de energia, quando cada nodo está ativo ou inativo** — esse é o
objetivo do código presente aqui.

Roadmap do projeto (curto → longo prazo):

1. **[Este repositório]** Identificação de padrões de consumo — classificar
   estados Ativo/Inativo de um nodo a partir do traço de potência, com um
   baseline heurístico e um modelo de ML (LSTM), comparados de forma direta.
2. IA na borda (Edge AI) — portar os modelos para microcontroladores
   (TensorFlow Lite for Microcontrollers, Edge Impulse).
3. Redes autônomas e heterogêneas — auto-gestão de duty-cycle e hierarquia
   de nodos (LoRa + MQTT).

## O que este repositório faz hoje

Classificação binária **DORMINDO (inativo) / ACORDADO (ativo)** a partir de um
traço de potência de um nodo sensor (`Delta-T_[ms]`, `Consumed_[mW]`), com:

- baseline heurístico (Savitzky-Golay + limiar);
- baseline trivial de referência (limiar de uma linha sobre a amostra);
- modelo LSTM;
- comparação direta entre os três, e avaliação da precisão temporal das
  transições detectadas (não só da classificação ponto a ponto).

## Estrutura

- `main.py` — pipeline completo (pré-processamento, treino, avaliação, gráficos).
- `test_main.py` — testes unitários das funções de detecção de ciclos e métricas.
- `requirements.txt` — dependências.

Não há diretórios `data/`, `notebooks/` ou `src/` neste momento — o dado usado
é externo ao repositório (ver "Como rodar" abaixo) e o pipeline ainda vive
num único módulo.

## Como rodar

```bash
pip install -r requirements.txt
python main.py --csv-path /caminho/para/seu.csv --output-dir ./resultados
```

Ou defina as variáveis de ambiente `MONDESA_CSV_PATH` e `MONDESA_OUTPUT_DIR`.

## Limitação conhecida

Os rótulos de treino (`Estado_Real`) são derivados de um limiar fixo (100 mW)
sobre o próprio sinal de potência que o modelo consome — não são uma medição
independente do estado do dispositivo. Isso significa que a avaliação atual
mede o quanto os modelos reproduzem essa regra, não necessariamente o
comportamento real de ativo/inativo do hardware. Uma fonte de rótulo
independente (datasheet do sensor, anotação manual, ou outro canal de medição)
é necessária antes de generalizar as conclusões.
