"""Junta os módulos do projeto num único .py autossuficiente para o Kaggle.

Uso: python gerar_arquivo_unico.py . mondesa_kaggle.py
"""
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1] if len(sys.argv) > 1 else "."); saida = Path(sys.argv[2] if len(sys.argv) > 2 else "mondesa_kaggle.py")
ORDEM = ['ciclos', 'dados', 'heuristico', 'avaliacao', 'rotulagem', 'modelos', 'lstm', 'main']
LOCAIS = '|'.join(ORDEM)

cabecalho = '''"""Projeto Mondesa — detecção de ciclos (t0, t1, t2) em séries de potência.

ARQUIVO ÚNICO para rodar no Kaggle: junta os módulos do repositório
(ciclos, dados, heuristico, avaliacao, rotulagem, modelos, lstm, main).
Gerado automaticamente — ao mudar o código, edite os módulos do repositório.

Uso no Kaggle (Internet ligada):
    !pip install -q hmmlearn ruptures
    !python mondesa_kaggle.py --csv-path /kaggle/input/.../experiment4-0.CSV --output-dir /kaggle/working/resultados

Os rótulos do chunk 12 (13 ciclos, transcritos da figura do coorientador)
estão embutidos em ROTULOS_EMBUTIDOS. Para usar outro arquivo: --rotulos caminho.csv
"""
'''
partes = []
imports_topo = set()
for nome in ORDEM:
    src = (repo / f'{nome}.py').read_text()
    src = re.sub(r'^"""(.*?)"""\n', '', src, count=1, flags=re.S)  # docstring do módulo
    # imports entre módulos do projeto (no topo ou dentro de funções) saem: tudo está no mesmo arquivo
    src = re.sub(rf'^[ \t]*from ({LOCAIS}) import \([^)]*\)\n', '', src, flags=re.M)
    src = re.sub(rf'^[ \t]*from ({LOCAIS}) import [^\n]*\n', '', src, flags=re.M)
    if nome == 'rotulagem':
        # a linha de comando própria da rotulagem não entra (o main é o do pipeline)
        src = src[:src.index('# ---------- linha de comando ----------')]
    if nome != 'main':
        src = re.sub(r"\nif __name__ == '__main__':\n    main\(\)\n?$", '\n', src)
    partes.append(f'\n# {"=" * 76}\n# {nome}.py\n# {"=" * 76}\n{src.strip()}\n')

codigo = cabecalho + '\n'.join(partes)

rotulos = (repo / 'rotulos' / 'rotulos_ciclos.csv').read_text().strip()
embutidos = f'\n\nROTULOS_EMBUTIDOS = """\\\n{rotulos}\n"""\n'
codigo = codigo.replace("\nMODELOS = [", embutidos + "\nMODELOS = [", 1)

# sem arquivo de rótulos, usa os embutidos (antes de cair nos provisórios)
antigo = "    ciclos = rotulos_provisorios(potencia, tempo, limiar_mw=100.0, min_duracao_s=MIN_DURACAO_S)\n"
novo = ("    if ROTULOS_EMBUTIDOS.strip():\n"
        "        import io\n"
        "        ciclos = carregar_rotulos(io.StringIO(ROTULOS_EMBUTIDOS))\n"
        "        print(f\"✓ {len(ciclos)} ciclos rotulados (embutidos no arquivo: chunk 12 do coorientador)\\n\")\n"
        "        return ciclos, 'arquivo'\n" + antigo)
assert codigo.count(antigo) == 1
codigo = codigo.replace(antigo, novo)


saida.write_text(codigo)
print(f'{saida}: {len(codigo.splitlines())} linhas')
