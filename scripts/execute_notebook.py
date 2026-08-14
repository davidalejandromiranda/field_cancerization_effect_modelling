"""Execute the supporting-information notebook in place and fail on errors."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


path = Path("simulation_fce_volume_results.ipynb")
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
)
client.execute()
nbformat.write(notebook, path)

errors = [
    output
    for cell in notebook.cells
    if cell.cell_type == "code"
    for output in cell.get("outputs", [])
    if output.output_type == "error"
]
if errors:
    raise RuntimeError(f"Notebook completed with {len(errors)} error output(s)")

print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells without errors")
