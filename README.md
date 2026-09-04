# RLII_26

Gemeinsames Repository für die universitäre Vorlesung **Reinforcement Learning II 2026**.

Dieses Repository basiert auf dem
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) und verwendet dessen
reale Experiment-Infrastruktur für Stable-Baselines3. Der Zoo-Code bleibt weitgehend
unverändert; kursbezogene Materialien werden schrittweise in `course/` ergänzt.

## Voraussetzungen

- Git
- `uv` zur Verwaltung von Python, virtueller Umgebung und Paketen
- `ffmpeg` für spätere Videoerzeugung

Python muss nicht separat installiert werden. `uv` installiert die für den Kurs
festgelegte Python-Version 3.12.

### `uv` einmalig installieren

Wenn `uv --version` bereits funktioniert, kann dieser Abschnitt übersprungen werden.

macOS mit Homebrew:

```bash
brew install uv
```

macOS oder Linux ohne Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Anschließend gegebenenfalls ein neues Terminal öffnen und prüfen:

```bash
uv --version
```

## Repository installieren

```bash
git clone https://github.com/Mannheim-Probability/RLII_26.git
cd RLII_26

uv python install 3.12
```

Abhängig vom Betriebssystem folgt der vollständige Paket-Sync.

macOS benötigt für den PyBullet-Quell-Build eine zusätzliche Compiler-Option:

```bash
CFLAGS="-fno-define-target-os-macros" uv sync --locked --all-extras
```

Linux und Windows:

```bash
uv sync --locked --all-extras
```

Der Sync erstellt automatisch `.venv`, installiert dieses Repository im
Editable Mode und richtet den vollständigen Kursumfang ein. Dazu gehören unter
anderem Gymnasium mit Box2D, Stable-Baselines3, MuJoCo, PyBullet, Plotting,
Video-Unterstützung und Weights & Biases.

Eine manuelle Aktivierung der virtuellen Umgebung ist nicht erforderlich. Befehle
werden mit `uv run --locked ...` in der Projektumgebung ausgeführt.

## Installation prüfen

Python und Lockfile:

```bash
uv run --locked python --version
uv lock --check
```

Zentrale Pakete:

```bash
uv run --locked python -c "import gymnasium, stable_baselines3, rl_zoo3, mujoco, pybullet, wandb; print('Imports OK')"
```

LunarLander/Box2D:

```bash
uv run --locked python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); obs, info = env.reset(seed=0); print(obs.shape, env.action_space); env.close()"
```

MuJoCo:

```bash
uv run --locked python -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); obs, info = env.reset(seed=0); print(obs.shape); env.close()"
```

PyBullet:

```bash
uv run --locked python -c "import gymnasium as gym; import rl_zoo3.import_envs; env = gym.make('HalfCheetahBulletEnv-v0'); obs, info = env.reset(seed=0); print(obs.shape); env.close()"
```

Auf macOS kann PyBullet dabei Warnungen zu mehrfach geladenen SDL-Klassen aus
OpenCV und Pygame ausgeben. Wenn der Befehl mit der Observation-Form endet und
keinen Fehlercode liefert, ist der Smoke Test erfolgreich.

Video-Werkzeug:

```bash
ffmpeg -version
```

## Kurzer PPO-Test

Dieser Lauf prüft nur die Trainingspipeline; er erzeugt noch keinen gut trainierten
Agenten:

```bash
uv run --locked python train.py \
  --algo ppo \
  --env LunarLander-v3 \
  --n-timesteps 512 \
  --seed 0 \
  --device cpu \
  --eval-freq -1 \
  -params n_envs:1 n_steps:128 batch_size:64 n_epochs:1
```

Trainingsausgaben werden unter `logs/` abgelegt und nicht mit Git versioniert.

## Weights & Biases

Das Python-Paket wird bei der vollständigen Installation eingerichtet. Ein Account
und Login werden erst benötigt, wenn W&B in einer späteren Vorlesung verwendet wird:

```bash
uv run --locked wandb login
```

API-Keys und andere Zugangsdaten dürfen niemals in dieses Repository committed
werden.

## Repository-Überblick

- `rl_zoo3/`: Experiment-Infrastruktur des RL Baselines3 Zoo
- `hyperparams/`: Zoo-Hyperparameter für Algorithmen und Environments
- `train.py`, `enjoy.py`: gut sichtbare Top-Level-Entrypoints
- `scripts/`: Plot-, Video- und Experimentwerkzeuge des Zoos
- `tests/`: Upstream-Tests und spätere Kurs-Tests
- `docs/`: Referenzdokumentation des Zoos
- `course/`: wird für Vorlesungen, Übungen und Setup-Anleitungen ergänzt

Das Submodule `rl-trained-agents` enthält Upstream-Modelle. Es wird für den normalen
Kurs-Workflow nicht benötigt und soll deshalb nicht initialisiert werden.

## Herkunft und Lizenz

Der Ausgangspunkt dieses Lehr-Forks ist
[`DLR-RM/rl-baselines3-zoo`](https://github.com/DLR-RM/rl-baselines3-zoo), Commit
`bef2e8fda66c792ee3ae733eaa6d22294b00737e`.

Der Upstream-Code steht unter der MIT-Lizenz; siehe [LICENSE](LICENSE).
