# Driving AI training

Launch the interactive game and online learner with:

```powershell
.\quantum\Scripts\python.exe .\car.py --mode gui
```

Run the shared-policy trainer without opening a window with:

```powershell
.\quantum\Scripts\python.exe .\car.py --mode headless
```

Headless training defaults to 1,000 lockstep simulations and 1,000 shared
policy updates. Each update waits for every environment in the batch to finish,
then trains once on their complete episodes. The default device is CUDA and
training stops with an actionable error if CUDA is unavailable; use
`--device cpu` to deliberately run on the CPU.

Useful controls:

```powershell
.\quantum\Scripts\python.exe .\car.py --mode headless --environments 64 --updates 10 --minibatch-size 2048
.\quantum\Scripts\python.exe .\car.py --mode headless --device cpu --environments 8 --updates 2
```

The learning rate decays by `0.995` per shared update to a floor of `1e-5`.
`--learning-rate-decay`, `--optimization-epochs`, `--report-every`, and
`--seed` can also be adjusted. Training resumes automatically from
`car_ai_shared_checkpoint_v1_run_*.pt`; use `--checkpoint` to select another
checkpoint family. These files are separate from the GUI's existing policy
checkpoints.

Verify CUDA in the configured environment with:

```powershell
.\quantum\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

For Python 3.12 on Windows, the configured environment uses the official CUDA
13.0 PyTorch wheel:

```powershell
.\quantum\Scripts\python.exe -m pip install --upgrade "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

If CUDA is unavailable, install a CUDA-enabled PyTorch build compatible with
the installed NVIDIA driver, then rerun the verification command.
