import importlib.metadata as m,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
versions={row['package']:m.version(row['package']) for row in json.loads((root/'sources/dependency-wheels.json').read_text())}
for row in json.loads((root/'sources/dependency-wheels.json').read_text()):
    assert versions[row['package']]==row['version'],versions
print(json.dumps({'isolated_dependency_versions':versions}))
