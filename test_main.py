"""
Testes do pipeline. Rodar com: pytest -v

Cobrem: utilitários de ciclos (debouncing, alternância, ciclos <-> estados),
rótulos (Fase 2), heurística (Fase 1), métricas de evento (Fase 5),
modelos (Fases 3, 4 e 6) e um teste de ponta a ponta na série sintética.
"""
import numpy as np
import pytest

from ciclos import (_calcular_cv_seguro, aplicar_debouncing_estados, detectar_transicoes,
                    extrair_ciclos, ciclos_para_estados, estados_para_ciclos, agrupar_ciclos_contiguos)
from dados import gerar_serie_sintetica
from heuristico import estimar_limiar_otsu, suavizar, detectar_ciclos_heuristico
from avaliacao import metricas_eventos, avaliar_modelo, eventos_reais
from rotulagem import carregar_rotulos, salvar_rotulos, validar_ciclos, rotulos_provisorios
from main import dividir_ciclos, indices_dos_ciclos, main


@pytest.fixture(scope='module')
def serie():
    _, tempo, potencia, ciclos = gerar_serie_sintetica(n_ciclos=10, seed=1)
    return tempo, potencia, ciclos


# ---------- _calcular_cv_seguro ----------

def test_cv_seguro_vetor_vazio():
    assert _calcular_cv_seguro(np.array([])) == 0.0


def test_cv_seguro_media_zero():
    assert _calcular_cv_seguro(np.array([0.0, 0.0, 0.0])) == 0.0


def test_cv_seguro_valor_normal():
    assert _calcular_cv_seguro(np.array([10.0, 10.0, 10.0])) == pytest.approx(0.0)


# ---------- debouncing / transições / ciclos ----------

def test_debouncing_remove_pulso_curto_e_preserva_alternancia():
    tempo = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    estados = np.array([1, 1, 0, 1, 1, 1])  # o "0" isolado dura só 1s
    limpo = aplicar_debouncing_estados(estados, tempo, min_duracao=2.0)
    assert np.sum(np.diff(limpo) != 0) == 0


def test_debouncing_mantem_pulso_longo():
    tempo = np.arange(10, dtype=float)
    estados = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    assert 0 in aplicar_debouncing_estados(estados, tempo, min_duracao=2.0)


def test_transicoes_e_ciclos_ficam_consistentes_apos_filtragem():
    tempo = np.linspace(0, 20, 200)
    sinal = ((tempo % 5) > 3).astype(int)
    _, tempos_trans, estados_trans = detectar_transicoes(sinal, tempo, min_duracao=0.5)
    for c in extrair_ciclos(tempos_trans, estados_trans):
        assert c['duracao_acordado'] == pytest.approx(c['t1'] - c['t0'])
        assert c['duracao_dormindo'] == pytest.approx(c['t2'] - c['t1'])
        assert c['t0'] < c['t1'] < c['t2']


def test_ciclos_para_estados_e_volta(serie):
    tempo, _, ciclos = serie
    estados, rotulado = ciclos_para_estados(tempo, ciclos)
    # fora dos ciclos rotulados nada conta como rotulado
    assert not rotulado[tempo < ciclos[0]['t0']].any()
    recuperados = estados_para_ciclos(estados, tempo)
    assert len(recuperados) == len(ciclos) - 1  # o último t2 não tem subida depois dentro dos rótulos
    for r, c in zip(recuperados, ciclos):
        assert r['t0'] == pytest.approx(c['t0']) and r['t1'] == pytest.approx(c['t1'])


def test_agrupar_ciclos_contiguos_separa_trechos():
    ciclos = [{'t0': 0, 't1': 1, 't2': 10}, {'t0': 10, 't1': 11, 't2': 20}, {'t0': 50, 't1': 51, 't2': 60}]
    assert [len(b) for b in agrupar_ciclos_contiguos(ciclos)] == [2, 1]


# ---------- divisão treino/val/teste ----------

def test_divisao_por_ciclo_e_cronologica():
    ciclos = [{'t0': 10.0 * i, 't1': 10.0 * i + 1, 't2': 10.0 * (i + 1)} for i in range(20)]
    tr, va, te = dividir_ciclos(ciclos)
    assert len(tr) + len(va) + len(te) == 20
    assert tr[-1]['t2'] <= va[0]['t0'] and va[-1]['t2'] <= te[0]['t0']
    with pytest.raises(ValueError):
        dividir_ciclos(ciclos[:4])


# ---------- Fase 2: rótulos ----------

def test_rotulos_ida_e_volta_e_filtro_revisado(tmp_path):
    ciclos = [{'ciclo_id': 1, 't0': 1.0, 't1': 2.0, 't2': 10.0}, {'ciclo_id': 2, 't0': 10.0, 't1': 11.0, 't2': 20.0}]
    caminho = salvar_rotulos(ciclos, str(tmp_path / 'r.csv'), revisado=1)
    assert len(carregar_rotulos(caminho)) == 2
    salvar_rotulos(ciclos, caminho, revisado=0)
    assert carregar_rotulos(caminho) == []  # não revisados são ignorados


def test_rotulos_invalidos_sao_recusados():
    with pytest.raises(ValueError):
        validar_ciclos([{'ciclo_id': 1, 't0': 5.0, 't1': 2.0, 't2': 10.0}])
    with pytest.raises(ValueError):
        validar_ciclos([{'ciclo_id': 1, 't0': 0.0, 't1': 1.0, 't2': 10.0},
                        {'ciclo_id': 2, 't0': 8.0, 't1': 9.0, 't2': 20.0}])


def test_rotulos_provisorios_vem_do_limiar_fixo(serie):
    tempo, potencia, ciclos = serie
    prov = rotulos_provisorios(potencia, tempo, limiar_mw=100.0)
    assert abs(len(prov) - (len(ciclos) - 1)) <= 1


# ---------- Fase 1: heurística ----------

def test_otsu_separa_modos_sem_usar_rotulos():
    rng = np.random.default_rng(0)
    potencia = np.r_[rng.normal(55, 3, 5000), rng.normal(300, 20, 500)]
    assert 70 < estimar_limiar_otsu(potencia) < 250


def test_suavizacao_nao_afunda_antes_do_degrau():
    """Regressão: o Savitzky-Golay descia a ~25 mW antes de cada subida."""
    tempo = np.arange(400) * 0.057
    potencia = np.where(np.arange(400) >= 200, 600.0, 55.0)
    potencia[200:210] = 700.0
    assert suavizar(potencia, tempo).min() >= 55.0 - 1e-9


def test_heuristica_devolve_t0_t1_t2(serie):
    tempo, potencia, ciclos = serie
    det, _ = detectar_ciclos_heuristico(potencia, tempo, estimar_limiar_otsu(potencia))
    assert len(det) == len(ciclos)  # o sinal tem um período ativo depois do último t2
    for d, c in zip(det, ciclos):
        for k in ('t0', 't1', 't2'):
            assert d[k] == pytest.approx(c[k], abs=0.2)


# ---------- Fase 5: métricas de evento ----------

def test_eventos_perfeitos():
    r = metricas_eventos([10.0, 20.0, 30.0], [10.0, 20.0, 30.0], tol_s=1.0)
    assert r['precisao'] == r['revocacao'] == 1.0 and r['mae_s'] == 0.0


def test_eventos_fora_da_tolerancia_nao_contam():
    r = metricas_eventos([10.5, 22.0], [10.0, 20.0], tol_s=1.0)
    assert r['tp'] == 1 and r['fp'] == 1 and r['fn'] == 1
    assert r['mae_s'] == pytest.approx(0.5)


def test_casamento_e_um_para_um():
    # dois previstos perto do mesmo real: só um pode ser acerto
    r = metricas_eventos([9.9, 10.2], [10.0], tol_s=1.0)
    assert r['tp'] == 1 and r['fp'] == 1 and r['mae_s'] == pytest.approx(0.1)


def test_eventos_reais_incluem_t2_do_ultimo_ciclo():
    ciclos = [{'t0': 0.0, 't1': 1.0, 't2': 10.0}, {'t0': 10.0, 't1': 11.0, 't2': 20.0}]
    subidas, descidas = eventos_reais(ciclos)
    assert list(subidas) == [0.0, 10.0, 20.0] and list(descidas) == [1.0, 11.0]


def test_avaliar_modelo_ignora_eventos_fora_dos_trechos_rotulados():
    ciclos = [{'t0': 100.0, 't1': 106.0, 't2': 400.0}]
    pred = [{'t0': 100.2, 't1': 106.1, 't2': 400.3}]
    r = avaliar_modelo([5.0, 100.2, 400.3], [11.0, 106.1], pred, ciclos, tol_s=1.0)
    assert r['precisao'] == 1.0 and r['revocacao'] == 1.0
    assert r['mae_t0_s'] == pytest.approx(0.2) and r['ciclos_recuperados'] == 1


# ---------- Fases 3, 4 e 6: modelos ----------

def test_hmm_ignora_picos_isolados_de_ruido(serie):
    from modelos import HMMDoisEstados
    tempo, potencia, ciclos = serie
    estados, _ = ciclos_para_estados(tempo, ciclos)
    idx = indices_dos_ciclos(tempo, ciclos[:6])
    est = HMMDoisEstados().treinar(potencia, tempo, estados, idx).prever(0, len(potencia))
    trans = np.sum(np.diff(est) != 0)
    assert trans <= 2 * (len(ciclos) + 1)


def test_classificador_janela_usa_so_o_passado():
    from modelos import calcular_features
    p = np.r_[np.full(100, 55.0), np.full(100, 160.0)]
    f1, f2 = calcular_features(p), calcular_features(np.r_[p[:150], np.full(50, 999.0)])
    # mudar o futuro (amostras >= 150) não muda as features até a amostra 149
    assert np.allclose(f1[:150], f2[:150])


def test_janela_lstm_termina_no_alvo():
    pytest.importorskip('tensorflow')
    from lstm import janelas
    serie_ = np.arange(20, dtype=np.float32)
    X, alvos = janelas(serie_, np.arange(20), seq_length=4)
    assert alvos[0] == 3  # alvos sem histórico suficiente são descartados
    assert np.all(X[:, -1, 0] == alvos) and np.all(X[:, 0, 0] == alvos - 3)


# ---------- ponta a ponta ----------

def test_pipeline_ponta_a_ponta_sem_lstm(tmp_path):
    tabela = main(['--csv-path', str(tmp_path / 'nao_existe.csv'), '--output-dir', str(tmp_path),
                   '--sintetico-ciclos', '15', '--modelos', 'limiar,heuristico,logreg,rf,hmm,ruptures'])
    assert len(tabela) == 6
    assert (tabela['f1'] > 0.9).all()
    assert (tmp_path / 'comparacao_modelos.csv').exists()
    assert len(list((tmp_path / 'pdf').glob('*.pdf'))) == 5
