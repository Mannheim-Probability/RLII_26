# Hard-Target TD3 v2 auf Helix

## Checkout und Python-Umgebung

Auf dem Helix-Login-Knoten mit `uv` auf dem `PATH` ausführen. Der Setup-Befehl verwendet die im Repo festgelegte Python-Version `3.12`, `uv.lock` und dieselben Extras wie die lokale Linux-Installationsanleitung. Er erzeugt eine Linux-`.venv` direkt im Checkout. [uv-Installationsanleitung](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone --branch daniel https://github.com/Mannheim-Probability/RLII_26.git
cd RLII_26
bash scripts/helix/setup.sh
```

In einem vorhandenen Checkout mit sauberem Arbeitsbaum zuerst `git fetch origin` und `git switch daniel` ausführen. Der Jobstarter verwendet standardmäßig `.venv/bin/python` oder den Interpreter einer aktivierten Virtualenv. Mit `HELIX_PYTHON=/absoluter/pfad/bin/python` lässt sich ein Interpreter explizit wählen. `HELIX_CUDA_MODULE` steht standardmäßig auf `devel/cuda`. Mit `HELIX_CUDA_MODULE=none` kann ein bereits geladener beziehungsweise vom PyTorch-Wheel mitgebrachter CUDA-Stack verwendet werden.

## GPU-Pilot und Laufzeitschätzung

Für W&B-Online-Logging auf Helix einmal `.venv/bin/wandb login` ausführen. Die Jobs starten mit `WANDB_MODE=online`. Bei fehlendem Netzzugang auf den GPU-Knoten kann der Pilot explizit im Offline-Modus laufen. Die Run-Verzeichnisse lassen sich anschließend vom Login-Knoten mit `.venv/bin/wandb sync /pfad/zum/offline-run-verzeichnis` übertragen.

Für einen ersten Lauf auf der Devel-Partition reicht je ein Seed pro Algorithmus. Die A40-Anforderung und das 30-Minuten-Limit kommen aus dem Skript.

```bash
sbatch --partition=devel --array=0,10 scripts/helix/td3_hard_targets_test.slurm
```

Für die vollständige Laufzeitmessung über alle zehn Seeds pro Algorithmus:

```bash
pilot=$(sbatch --parsable scripts/helix/td3_hard_targets_test.slurm)
squeue -j "$pilot"
sacct -j "$pilot" --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

Der Pilot startet jeweils zehn Seeds für `td3_target_interval` und `ic_td3` mit (K=1.000), Adam und 500.000 Environment-Schritten. Das Zeitlimit beträgt 30 Minuten pro Task. Für einen kürzeren Pilot lässt sich beim `sbatch`-Aufruf nach dem Skriptnamen beispielsweise `--pilot-steps 250000` setzen. Er verwendet dieselbe Architektur, Lernrate und Evaluationsfrequenz wie der vollständige Sweep. Als separater SGD-Pilot ist `sbatch scripts/helix/td3_hard_targets_test.slurm --pilot-optimizer sgd` möglich.

Jeder Pilot-Task prüft den CUDA-Zugriff mit einer Matrixmultiplikation, die Python-Abhängigkeiten und einen LunarLander-Schritt. Nach dem Training prüft er den Ziel-Update-Zähler, die gespeicherten Netzparameter, geloggte Actor- und Critic-Losses, alle planmäßigen Evaluationen und die lokale W&B-Run-Datei. Ein erfolgreicher Task schreibt eine `pilot_result.json` mit Durchsatz und geschätzter Dauer bis fünf Millionen Schritte. Die Schätzung verwendet die Zeit zwischen der ersten und der letzten Evaluation.

```bash
find "${HELIX_OUTPUT_ROOT:-logs/target_updates_td3/helix-v2}/pilot/$pilot" \
  -name pilot_result.json -exec grep -H projected_full_hours {} +
```

## Vollständiger Sweep

Vor der Einreichung die Pilot-Logs und Laufzeitschätzungen prüfen. Das Full-Array hat vorläufig 48 Stunden je Task. Die Zeit kann bei der Einreichung per `--time` an die beobachteten Laufzeiten angepasst werden. Beide Arrays verwenden `gpu-single`, eine A40 pro Task, vier CPU-Kerne und 16 GB RAM. Auf Helix beträgt die maximale Laufzeit pro `gpu-single`-Job 120 Stunden. Falls deine Zuordnung einen expliziten Account braucht, ergänze beim jeweiligen `sbatch`-Aufruf `--account=…`. [Helix Slurm](https://wiki.bwhpc.de/e/Helix/Slurm#Partitions)

```bash
sbatch scripts/helix/td3_hard_targets_full.slurm
```

Die 200 Tasks decken beide Algorithmen, Adam und SGD ohne Momentum, (K\in\{2,100,1.000,10.000,50.000\}) und jeweils die Seeds `0–9` ab. Pro Task läuft ein Seed mit fünf Millionen Environment-Schritten, maximal fünf Tasks laufen gleichzeitig. Jedes Training verwendet `tau=1.0`, `gamma=0.99`, `policy_delay=2` und `learning_rate=0.001`. Evaluiert wird alle 25.000 Environment-Schritte mit fünf Episoden.

Alle vollständigen Läufe loggen in `RL2_2026/target_updates_td3` mit genau dem Tag `td3-hard-targets-adam-sgd-v2`. Die Pilotläufe erhalten den separaten Tag `td3-hard-targets-adam-sgd-v2-pilot`. Ergebnisse landen standardmäßig in `logs/target_updates_td3/helix-v2`, mit `HELIX_OUTPUT_ROOT` lässt sich ein anderer Pfad wählen. Die Slurm-Logs `slurm-td3-v2-*_*.out` liegen im Einreichungsverzeichnis.

Die Zuordnung einzelner Array-Tasks kann ohne GPU geprüft werden:

```bash
.venv/bin/python -m scripts.helix.td3_hard_targets full --task-id 199 --dry-run
```
