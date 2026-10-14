# Apex Driver Lab

A procedural driving simulator and reinforcement-learning sandbox built with Python and PyTorch. Train a policy against generated roads and traffic, inspect its behavior in a live GUI, and resume from versioned local checkpoints.

> This project is an educational simulation. It is not designed, validated, or safe for controlling real vehicles.

## Capabilities

- Deterministic procedural road generation with streamed track sections.
- Traffic simulation with proximity sensing, predictive braking, and collision detection based on vehicle contact.
- Smooth steering, lane-centering assistance, and curve-aware speed targets.
- Interactive Tkinter visualization with live training and safety telemetry.
- Batched headless training with a shared PPO policy, resumable checkpoints, and continuous mode.
- Automated tests for simulation behavior, training, checkpoint migration, and GUI rendering.

## Quick start

Python 3.12 is the primary development and test environment. Install the project and test dependencies in a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[test]"
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Launch the simulation:

```powershell
python .\car.py --mode gui
```

On Linux systems without Tkinter installed, install the operating system's `python3-tk` package.

## Train

Run a bounded CPU training session:

```powershell
python .\car.py --mode headless --device cpu --environments 32 --updates 5
```

Train until manually stopped with `Ctrl+C`:

```powershell
python .\car.py --mode headless --device cpu --environments 32 --until-stopped
```

Headless training defaults to CUDA and 1,000 environments/updates. Select `--device cpu` when CUDA is unavailable. Each completed update is checkpointed. The GUI chooses the most recently saved compatible checkpoint among the GUI and shared headless training families, and reports the source and progress in its status bar.

See [TRAINING.md](TRAINING.md) for options, checkpoint behavior, and CUDA setup.

## Test

```powershell
python -m pytest -q
```

The GitHub Actions workflow runs the test suite on Windows and Ubuntu with Python 3.12.

## Project layout

| Path | Responsibility |
| --- | --- |
| `car.py` | Procedural track, Tkinter GUI, and command-line entry point |
| `car_simulation.py` | Vehicle dynamics, observations, traffic, safety, rewards, and collisions |
| `car_ai.py` | Policy networks, online learning, shared PPO training, and checkpoints |
| `car_training.py` | Batched headless training orchestration |
| `tests/` | Simulation, policy, checkpoint, CLI, and GUI tests |
| `TRAINING.md` | Training configuration and operational guidance |

## Model files

Training checkpoints are machine-local artifacts. They are intentionally excluded from Git; keep and back them up separately if they are important. Use `--checkpoint PATH` to select a checkpoint family outside the repository.

## Contributing

Bug reports, focused improvements, and documentation fixes are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Security issues should follow [SECURITY.md](SECURITY.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for the full terms.
