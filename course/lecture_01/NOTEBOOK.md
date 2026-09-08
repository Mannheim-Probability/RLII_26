# Lecture 01 notebook

Open `lecture_01.ipynb` in JupyterLab or an editor with notebook support.
The teaching content is in English.

## Start with the project environment

From the repository root, install the project as described in the main README.
`uv sync --locked --all-extras` includes JupyterLab and ipykernel from the
`notebooks` extra, with versions resolved in `uv.lock`.

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Launch JupyterLab with that interpreter:

```bash
python -m jupyterlab course/lecture_01/lecture_01.ipynb
```

In VS Code, open the notebook and select the repository's `.venv` interpreter
using **Select Kernel**. The first cell prints its executable path and installed
package versions so the selected environment can be verified.

## Teaching workflow

- Follow the source navigation and environment examples in the notebook.
- Adapt `scripts/train_ppo_example.sh` for the assignment described there.
- Run each chosen seed sequentially with the project environment activated.
- Execute the final notebook cells to generate one mean evaluation return plot with a shaded standard error from your own completed runs.
- The optional VS Code debugger uses a separate short example; see `DEBUGGING.md`.
- The introductory video is included in the repository and displayed by the notebook.
- Library links use the macOS/Linux `.venv` layout; Windows uses `.venv/Lib/site-packages/`.
- Save or close the notebook before external edits; clear all outputs before committing.
