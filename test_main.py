"""
Testes mínimos para main.py — cobrindo os pontos que a revisão de
orientação (d26600c) apontou como sensíveis: CV protegido contra
divisão por zero, alternância de estados após debouncing, e a
métrica de avaliação de transições.

Rodar com: pytest test_main.py -v
"""
import numpy as np
import pytest

from main import (
    _calcular_cv_seguro,
    aplicar_debouncing_estados,
    detectar_transicoes,
    extrair_ciclos,
    avaliar_transicoes,
    dividir_dados_brutos_cronologico,
    rotular_estados,
    estimar_limiar_otsu,
    preparar_sequencias,
    avaliar_baseline_trivial,
)


# ---------- _calcular_cv_seguro ----------

def test_cv_seguro_vetor_vazio():
    assert _calcular_cv_seguro(np.array([])) == 0.0


def test_cv_seguro_media_zero():
    # média zero causaria ZeroDivisionError/inf sem a proteção
    assert _calcular_cv_seguro(np.array([0.0, 0.0, 0.0])) == 0.0


def test_cv_seguro_valor_normal():
    vetor = np.array([10.0, 10.0, 10.0])
    assert _calcular_cv_seguro(vetor) == pytest.approx(0.0)


# ---------- aplicar_debouncing_estados ----------

def test_debouncing_remove_pulso_curto_e_preserva_alternancia():
    # pulso de 1 -> 0 -> 1 com duração menor que min_duracao no meio
    tempo = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    estados = np.array([1, 1, 0, 1, 1, 1])  # o "0" isolado dura só 1s
    limpo = aplicar_debouncing_estados(estados, tempo, min_duracao=2.0)
    # não deve sobrar um estado isolado quebrando a alternância
    transicoes = np.sum(np.diff(limpo) != 0)
    assert transicoes == 0  # o pulso curto deve ser absorvido


def test_debouncing_mantem_pulso_longo():
    tempo = np.arange(10, dtype=float)
    estados = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])  # bloco "0" dura 5s
    limpo = aplicar_debouncing_estados(estados, tempo, min_duracao=2.0)
    assert 0 in limpo  # o bloco longo deve sobreviver


# ---------- detectar_transicoes / extrair_ciclos ----------

def test_transicoes_e_ciclos_ficam_consistentes_apos_filtragem():
    """
    Regressão para o bug 'filtrar transições curtas quebra a
    alternância e fabrica ciclos' (main.py:364-374 na revisão original).
    As durações de cada ciclo extraído devem bater com a diferença
    real entre os tempos de transição retornados.
    """
    tempo = np.linspace(0, 20, 200)
    sinal = ((tempo % 5) > 3).astype(int)  # pulsos regulares
    _, tempos_trans, estados_trans = detectar_transicoes(sinal, tempo, min_duracao=0.5)
    ciclos = extrair_ciclos(tempos_trans, estados_trans)

    for c in ciclos:
        assert c['duracao_acordado'] == pytest.approx(c['t1'] - c['t0'])
        assert c['duracao_dormindo'] == pytest.approx(c['t2'] - c['t1'])
        # as três transições de um ciclo devem estar em ordem estrita
        assert c['t0'] < c['t1'] < c['t2']


# ---------- avaliar_transicoes ----------

def test_avaliar_transicoes_perfeito():
    tempos = [10.0, 20.0, 30.0]
    r = avaliar_transicoes(pred_times=tempos, true_times=tempos, tolerance_s=1.0)
    assert r['precision'] == 1.0
    assert r['recall'] == 1.0
    assert r['mae_s'] == 0.0


def test_avaliar_transicoes_usa_t0_e_t1_do_heuristico():
    """
    Regressão para o bug encontrado na revisão real: usar só 't0'
    de cada ciclo heurístico trava artificialmente o recall em 50%,
    porque as transições reais alternam liga/desliga.
    """
    ciclos_heur = [
        {'t0': 10.0, 't1': 15.0},
        {'t0': 30.0, 't1': 35.0},
    ]
    tempos_todos = sorted([c['t0'] for c in ciclos_heur] + [c['t1'] for c in ciclos_heur])
    tempos_reais = [10.0, 15.0, 30.0, 35.0]

    r_completo = avaliar_transicoes(tempos_todos, tempos_reais, tolerance_s=1.0)
    assert r_completo['recall'] == 1.0

    # a versão com bug (só t0) não deveria passar de 50% de recall
    tempos_so_t0 = [c['t0'] for c in ciclos_heur]
    r_bugado = avaliar_transicoes(tempos_so_t0, tempos_reais, tolerance_s=1.0)
    assert r_bugado['recall'] == pytest.approx(0.5)


# ---------- dividir_dados_brutos_cronologico ----------

def test_divisao_e_cronologica_e_nao_embaralha():
    n = 100
    potencia = np.arange(n, dtype=float)
    estados = np.zeros(n, dtype=int)
    tempo = np.arange(n, dtype=float)

    idx_tr, idx_va, idx_te = dividir_dados_brutos_cronologico(potencia, estados, tempo)

    assert idx_tr.max() < idx_va.min()
    assert idx_va.max() < idx_te.min()
    assert len(idx_tr) + len(idx_va) + len(idx_te) == n

# ---------- rótulos x baselines (circularidade) ----------

def test_otsu_separa_modos_sem_usar_rotulos():
    rng = np.random.default_rng(0)
    potencia = np.r_[rng.normal(55, 3, 5000), rng.normal(300, 20, 500)]
    limiar = estimar_limiar_otsu(potencia)
    assert 70 < limiar < 250


def test_preparar_sequencias_alvo_fora_da_janela():
    dados = np.arange(20, dtype=float)
    for h in (1, 5):
        X, y, idx = preparar_sequencias(dados, dados, seq_length=4, horizonte=h)
        # alvo = última amostra da janela + h
        assert np.all(y == X[:, -1, 0] + h)
        assert np.all(idx == y.astype(int))
    with pytest.raises(ValueError):
        preparar_sequencias(dados, dados, seq_length=4, horizonte=0)


def test_baseline_trivial_nao_e_a_regra_de_rotulagem():
    """
    Regressão para a revisão (áudio): o baseline usava x[t] > 100 mW,
    a mesma regra que gera Estado_Real, então acertava 100% por construção.
    O baseline atual usa x[t-h] (o que o LSTM vê) e erra nas transições.
    """
    tempo = np.arange(400, dtype=float)
    potencia = np.where((tempo % 100) >= 90, 300.0, 55.0)
    estados = rotular_estados(potencia, 100.0)
    X, y, idx = preparar_sequencias(potencia, estados, seq_length=10, horizonte=1)

    regra_do_rotulo = (potencia[idx] > 100.0).astype(int)
    assert np.array_equal(regra_do_rotulo, y)  # circular: 100% garantido

    m = avaliar_baseline_trivial(potencia[idx - 1], y, threshold=estimar_limiar_otsu(potencia))
    assert m['accuracy'] < 1.0
