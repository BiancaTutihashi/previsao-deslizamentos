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
from main import dividir_ciclos, indices_dos_ciclos, main, remover_sem_dependencia


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


def test_convencao_t1_e_a_ultima_amostra_acordada():
    """Mesma convenção dos rótulos do coorientador (conferida no experiment4-0):
    t0 = 1ª amostra acordada, t1 = ÚLTIMA amostra acordada, t2 = 1ª acordada do ciclo seguinte."""
    tempo = np.arange(12) * 0.05
    potencia = np.array([52, 52, 353, 161, 161, 52, 52, 52, 52, 340, 161, 52], dtype=float)
    ciclos = estados_para_ciclos((potencia > 100).astype(int), tempo, min_duracao=0.0)
    assert ciclos[0]['t0'] == tempo[2] and ciclos[0]['t1'] == tempo[4] and ciclos[0]['t2'] == tempo[9]
    estados, _ = ciclos_para_estados(tempo, ciclos)
    assert list(estados[2:5]) == [1, 1, 1] and estados[5] == 0  # [t0, t1] inclusive


def test_hmm_tolera_potencia_quantizada():
    """Regressão: no dado real o sono só vale 45,5/51,9/58,4 mW; o EM colapsava a variância
    e 45 ou 58 mW viravam 'acordado'."""
    from modelos import HMMDoisEstados
    rng = np.random.default_rng(0)
    tempo = np.arange(60000) * 0.05
    # treino: sono quase sempre 51,9 (o filtro de mediana apaga os desvios isolados);
    # depois: sequências de 58,4 mW que sobrevivem ao filtro, como no trecho de teste real
    potencia = np.full(len(tempo), 51.9)
    for i in rng.choice(np.arange(30000, 60000 - 5), size=200, replace=False):
        potencia[i:i + 6] = 58.4
    for t0 in np.arange(100, 2900, 300):
        # acordado: platô de ~161 mW com rajadas de 250–350 mW (gaussiana larga)
        i = np.searchsorted(tempo, t0); potencia[i:i + 130] = rng.choice([161.0, 250.0, 350.0], 130, p=[.6, .2, .2])
    estados = (potencia > 100).astype(int)
    hmm = HMMDoisEstados().treinar(potencia, tempo, estados, np.arange(30000))
    pred = hmm.prever(30000, len(tempo))
    assert np.sum(np.diff(pred) != 0) == np.sum(np.diff(estados[30000:]) != 0)


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


def test_modelo_sem_biblioteca_e_pulado_com_aviso(monkeypatch, capsys):
    import importlib.util
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda nome, *a, **k: None if nome == 'hmmlearn' else original(nome, *a, **k))
    assert remover_sem_dependencia(['heuristico', 'hmm', 'rf']) == ['heuristico', 'rf']
    assert 'pip install -q hmmlearn' in capsys.readouterr().out


def test_carrega_formato_bruto_do_registrador(tmp_path):
    """Formato dos exp8/11/12/15: 4 colunas inteiras sem cabeçalho (Δt ms, V µV, I nA invertida, ?)."""
    from dados import carregar_csv
    caminho = tmp_path / 'bruto.csv'
    caminho.write_text('0000000198,0005064268,-0010270141,0000025931\n'
                       '47,5064268,-10270141,25931\n'
                       '48,0,-10270141,25931\n'              # tensão 0: descartada
                       '47,5064268,2147483648,25931\n'       # corrente saturada: descartada
                       '52,5064268,-100000000,25931\n')
    df, tempo, potencia = carregar_csv(str(caminho))
    assert len(potencia) == 3  # a 1ª linha (lida como cabeçalho) volta a ser amostra
    assert potencia[0] == pytest.approx(52.01, abs=0.01)  # 5,064 V × 10,27 mA
    assert potencia[2] == pytest.approx(506.4, abs=0.1)
    assert tempo[0] == pytest.approx(0.198) and tempo[-1] == pytest.approx(0.198 + 0.047 + 0.048 + 0.047 + 0.052)


# ---------- rajadas, rótulos do rastreamento e regular × falho ----------

def test_limpar_estados_junta_rajada_e_descarta_pulso_curto():
    """Rajada do Wi-Fi: pulsos de 0,1 s com 0,3 s entre eles viram UM período ativo.
    O debouncing antigo apagava cada pulso curto antes de juntar, e a rajada sumia."""
    from ciclos import limpar_estados
    tempo = np.arange(0, 60, 0.05)
    e = np.zeros(len(tempo), dtype=int)
    for ini in np.arange(10, 13, 0.4):                       # rajada de 10 a ~13 s
        e[(tempo >= ini) & (tempo < ini + 0.1)] = 1
    e[(tempo >= 30) & (tempo < 30.1)] = 1                     # pulso isolado: não é ciclo
    e[(tempo >= 40) & (tempo < 42)] = 1; e[(tempo >= 43) & (tempo < 45)] = 1  # lacuna de 1 s: dois trechos
    limpo = limpar_estados(e, tempo, lacuna_s=0.5, min_ativo_s=1.0)
    assert np.sum(np.diff(limpo) == 1) == 3                   # rajada + dois trechos de 2 s
    assert limpo[(tempo > 10.05) & (tempo < 12.8)].all()      # a rajada ficou inteira
    assert not limpo[(tempo >= 29) & (tempo <= 31)].any()     # o pulso isolado sumiu


def test_carrega_rotulos_do_rastreamento(tmp_path):
    caminho = tmp_path / 'r.csv'
    caminho.write_text('k,t0,t1,t2,estado\n0,10,17,310,regular\n1,310,314,610,falho\n4,1200,1207,1500,regular\n')
    c = carregar_rotulos(str(caminho))
    assert [x['ciclo_id'] for x in c] == [0, 1, 4] and [x['estado'] for x in c] == ['regular', 'falho', 'regular']
    assert [len(b) for b in agrupar_ciclos_contiguos(c)] == [2, 1]   # k pulados = buraco entre blocos
    caminho.write_text('k,t0,t1,t2,estado\n0,10,17,310,quebrado\n')
    with pytest.raises(ValueError):
        carregar_rotulos(str(caminho))


def test_medidas_e_metricas_de_regular_falho():
    from ciclos import medidas_ciclo
    from avaliacao import avaliar_estados
    tempo = np.arange(0, 20, 1.0)
    potencia = np.full(20, 10.0); potencia[2:6] = 100.0; potencia[5] = 300.0
    m = medidas_ciclo({'t0': 2.0, 't1': 5.0, 't2': 12.0}, tempo, potencia)
    assert m['T_on'] == 3 and m['T'] == 10 and m['P_on'] == pytest.approx(150.0)
    assert m['P_off'] == pytest.approx(10.0) and m['P_max'] == 300.0
    reais = [{'t0': 0, 'estado': 'regular'}, {'t0': 300, 'estado': 'falho'},
             {'t0': 600, 'estado': 'falho'}, {'t0': 900, 'estado': 'regular'}]
    previstos = [{'t0': 0.2, 'estado': 'falho'}, {'t0': 300.1, 'estado': 'falho'}, {'t0': 900, 'estado': 'regular'}]
    r = avaliar_estados(previstos, reais, tol_s=1.0)   # o falho de 600 s nem foi detectado
    assert r['falhos_reais'] == 2 and r['falhos_detectados'] == 1
    assert r['precisao_falho'] == pytest.approx(0.5) and r['revocacao_falho'] == pytest.approx(0.5)
    assert r['acuracia_estado'] == pytest.approx(2 / 3)


def test_classificador_de_ciclos_com_uma_classe_so():
    from modelos import ClassificadorCiclos
    medidas = [{'T_on': 6.6, 'T': 306, 'P_on': 160, 'P_off': 52, 'P_max': 700}] * 5
    clf = ClassificadorCiclos().treinar(medidas, ['regular'] * 5)
    assert list(clf.prever(medidas[:2])) == ['regular', 'regular']
