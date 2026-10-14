"""Junta os módulos do projeto num único .py autossuficiente para o Kaggle.

Uso:
    python gerar_arquivo_unico.py                 # completo -> mondesa_kaggle.py
    python gerar_arquivo_unico.py --enxuto        # enxuto   -> mondesa_kaggle_enxuto.py

A versão enxuta tira o que nunca roda no Kaggle (série sintética, rótulos
provisórios, gráficos HTML) e as docstrings. Exige o CSV real e usa os
rótulos embutidos (ou --rotulos).
"""
import argparse
import ast
import re
from pathlib import Path

ORDEM = ['ciclos', 'dados', 'heuristico', 'avaliacao', 'rotulagem', 'modelos', 'lstm', 'main']
# funções que não entram nunca (linha de comando própria da rotulagem)
SEMPRE_FORA = {'rotulagem': {'_carregar_serie', 'main'}}
# funções que só servem fora do Kaggle (sem CSV ou sem arquivo de rótulos)
FORA_ENXUTO = {'dados': {'gerar_serie_sintetica', 'salvar_eda_html'},
               'rotulagem': {'rotulos_provisorios', 'salvar_rotulos', 'salvar_revisao_html'}}

OBTER_ROTULOS_ENXUTO = '''def obter_rotulos(args, tempo, potencia, ciclos_sinteticos):
    secao("FASE 2: RÓTULOS t0/t1/t2")
    if args.rotulos and os.path.exists(args.rotulos):
        ciclos, fonte = carregar_rotulos(args.rotulos), args.rotulos
    else:
        ciclos, fonte = carregar_rotulos(io.StringIO(ROTULOS_EMBUTIDOS)), 'embutidos (chunk 12 do coorientador)'
    print(f"✓ {len(ciclos)} ciclos rotulados: {fonte}\\n")
    return ciclos, 'arquivo'
'''

# (trecho original, trecho novo) aplicados ao main.py; cada um precisa existir exatamente uma vez
TROCAS_COMPLETO = [
    ("    ciclos = rotulos_provisorios(potencia, tempo, limiar_mw=100.0, min_duracao_s=MIN_DURACAO_S)\n",
     "    if ROTULOS_EMBUTIDOS.strip():\n"
     "        ciclos = carregar_rotulos(io.StringIO(ROTULOS_EMBUTIDOS))\n"
     "        print(f\"✓ {len(ciclos)} ciclos rotulados (embutidos no arquivo: chunk 12 do coorientador)\\n\")\n"
     "        return ciclos, 'arquivo'\n"
     "    ciclos = rotulos_provisorios(potencia, tempo, limiar_mw=100.0, min_duracao_s=MIN_DURACAO_S)\n"),
]
TROCAS_ENXUTO = [
    ("    if os.path.exists(args.csv_path):\n"
     "        df, tempo, potencia = carregar_csv(args.csv_path)\n"
     "    else:\n"
     "        print(f\"⚠️ {args.csv_path} não encontrado — usando série sintética com gabarito conhecido.\")\n"
     "        df, tempo, potencia, ciclos_sint = gerar_serie_sintetica(n_ciclos=args.sintetico_ciclos)\n",
     "    if not os.path.exists(args.csv_path):\n"
     "        raise SystemExit(f\"CSV não encontrado: {args.csv_path}\")\n"
     "    df, tempo, potencia = carregar_csv(args.csv_path)\n"),
    ("    salvar_eda_html(tempo, potencia, os.path.join(args.output_dir, 'html', 'eda_interativo.html'))\n", ""),
    ("    p.add_argument('--sintetico-ciclos', type=int, default=40,\n"
     "                   help='nº de ciclos da série sintética usada quando o CSV não existe')\n", ""),
    ("    p.add_argument('--rotulos', default=os.environ.get('MONDESA_ROTULOS', 'rotulos/rotulos_ciclos.csv'),\n"
     "                   help='CSV ciclo_id,t0,t1,t2 com os rótulos verdadeiros (Fase 2)')\n",
     "    p.add_argument('--rotulos', default='', help='CSV ciclo_id,t0,t1,t2 (padrão: os rótulos embutidos)')\n"),
    ("    if origem == 'provisorio':\n"
     "        print(\"\\n⚠️ Lembrete: rótulos PROVISÓRIOS (limiar 100 mW). Os números acima não valem como resultado.\")\n",
     ""),
]


def _docstring(no):
    corpo = getattr(no, 'body', None)
    if corpo and isinstance(corpo[0], ast.Expr) and isinstance(getattr(corpo[0], 'value', None), ast.Constant) \
            and isinstance(corpo[0].value.value, str):
        return corpo[0]
    return None


def processar(nome, src, enxuto, imports):
    """Remove imports (os de fora vão para `imports`), funções descartadas e docstrings."""
    arvore = ast.parse(src)
    linhas = src.splitlines(keepends=True)
    apagar, substituir = set(), {}
    fora = SEMPRE_FORA.get(nome, set()) | (FORA_ENXUTO.get(nome, set()) if enxuto else set())

    def intervalo(no):
        return range(no.lineno - 1, no.end_lineno)

    doc = _docstring(arvore)
    if doc:
        apagar.update(intervalo(doc))
    for no in arvore.body:
        if isinstance(no, (ast.Import, ast.ImportFrom)):
            if not (isinstance(no, ast.ImportFrom) and no.module in ORDEM):
                imports.append(ast.get_source_segment(src, no))
            apagar.update(intervalo(no))
        elif isinstance(no, ast.Expr) and 'matplotlib.use' in ast.get_source_segment(src, no):
            apagar.update(intervalo(no))
        elif isinstance(no, ast.If) and '__name__' in ast.get_source_segment(src, no.test) and nome != 'main':
            apagar.update(intervalo(no))
        elif isinstance(no, ast.FunctionDef) and no.name in fora:
            inicio = min([no.lineno] + [d.lineno for d in no.decorator_list]) - 1
            apagar.update(range(inicio, no.end_lineno))
        elif enxuto and nome == 'main' and isinstance(no, ast.FunctionDef) and no.name == 'obter_rotulos':
            substituir[no.lineno - 1] = OBTER_ROTULOS_ENXUTO
            apagar.update(range(no.lineno, no.end_lineno))
    for no in ast.walk(arvore):
        # imports de módulos do projeto dentro de funções: tudo está no mesmo arquivo agora
        if isinstance(no, ast.ImportFrom) and no.module in ORDEM:
            apagar.update(intervalo(no))
        if enxuto and isinstance(no, (ast.FunctionDef, ast.ClassDef)):
            d = _docstring(no)
            if d and len(no.body) > 1:
                apagar.update(intervalo(d))

    saida = []
    for i, linha in enumerate(linhas):
        if i in substituir:
            saida.append(substituir[i])
        elif i not in apagar:
            if enxuto and linha.lstrip().startswith('# ---------- '):
                continue
            saida.append(linha)
    return ''.join(saida)


def gerar(repo, enxuto):
    imports, partes = [], []
    for nome in ORDEM:
        corpo = processar(nome, (repo / f'{nome}.py').read_text(), enxuto, imports)
        corpo = re.sub(r'\n{3,}', '\n\n\n', corpo.strip('\n'))
        partes.append(f'# {"=" * 30} {nome}.py {"=" * 30}\n{corpo}\n')

    # imports únicos no topo; o backend do matplotlib tem de ser escolhido antes do pyplot
    vistos, topo = set(), []
    for imp in imports + ['import io']:
        if imp in vistos:
            continue
        vistos.add(imp)
        if imp == 'import matplotlib.pyplot as plt':
            topo.append("matplotlib.use('Agg')")
        topo.append(imp)

    codigo = '\n\n\n'.join(partes)
    trocas = TROCAS_ENXUTO if enxuto else TROCAS_COMPLETO
    for antigo, novo in trocas:
        if codigo.count(antigo) != 1:
            raise RuntimeError(f"trecho esperado não encontrado (o main.py mudou?):\n{antigo}")
        codigo = codigo.replace(antigo, novo)

    rotulos = (repo / 'rotulos' / 'rotulos_ciclos.csv').read_text().strip()
    codigo = codigo.replace('\nMODELOS = [', f'\nROTULOS_EMBUTIDOS = """\\\n{rotulos}\n"""\n\nMODELOS = [', 1)

    versao = 'ENXUTA' if enxuto else 'COMPLETA'
    cabecalho = f'''"""Projeto Mondesa — detecção de ciclos (t0, t1, t2) em séries de potência. Versão {versao} para o Kaggle.

Gerado por gerar_arquivo_unico.py a partir dos módulos do repositório: para mudar o
código, edite os módulos e gere de novo. Rótulos do chunk 12 embutidos (ROTULOS_EMBUTIDOS).

    !pip install -q hmmlearn ruptures
    !python {'mondesa_kaggle_enxuto.py' if enxuto else 'mondesa_kaggle.py'} --csv-path /kaggle/input/.../experiment4-0.CSV --output-dir /kaggle/working/resultados
"""
'''
    return cabecalho + '\n'.join(topo) + '\n\n\n' + codigo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--enxuto', action='store_true')
    p.add_argument('--repo', default='.')
    p.add_argument('--saida', default=None)
    args = p.parse_args()
    saida = Path(args.saida or ('mondesa_kaggle_enxuto.py' if args.enxuto else 'mondesa_kaggle.py'))
    codigo = gerar(Path(args.repo), args.enxuto)
    compile(codigo, str(saida), 'exec')
    saida.write_text(codigo)
    print(f'✓ {saida}: {len(codigo.splitlines())} linhas')


if __name__ == '__main__':
    main()
