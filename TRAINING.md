# Driving AI training

Launch the interactive game and online learner with:

```powershell
.\quantum\Scripts\python.exe .\car.py --mode gui
```

Run the shared-policy trainer without opening a window with:

```powershell
.\quantum\Scripts\python.exe .\car.py --mode headless
```

Headless training defaults to 1,000 concurrent simulations and 1,000 updates.
Without `--async_reset`, it uses the legacy completed-episode learner. With
`--async_reset`, it resets each simulation immediately after a terminal step
and collects only complete episodes in a shared buffer. It updates after the
buffer reaches either the configured completed-episode threshold or transition
threshold (defaults: 32 episodes / 8,192 transitions), and has at least 128
fresh complete episodes after stale-data filtering. Partial episodes are not
used for gradients and are discarded at shutdown.

The default device is CUDA and training stops with an actionable error if CUDA
is unavailable; use `--device cpu` to deliberately run on the CPU.

Useful controls:

```powershell
.\quantum\Scripts\python.exe .\car.py --mode headless --environments 64 --updates 10 --minibatch-size 2048
.\quantum\Scripts\python.exe .\car.py --mode headless --device cpu --environments 8 --updates 2
.\quantum\Scripts\python.exe .\car.py --mode headless --async_reset --environments 1000 --updates 100 --async-episodes 32 --async-transitions 8192 --max-policy-lag 4 --min_batch_episodes 128 --max-episode-steps 1800
```

The async collector normalizes discounted Monte Carlo returns over the entire
learner batch (not episode-by-episode) and applies the existing clipped PPO
ratio using stored behavior log-probabilities. It tags every action with its
policy update version per transition and drops a complete episode if any
transition is older than `--max-policy-lag` when an update is formed. Keeping
or dropping full episodes preserves their complete Monte Carlo returns.
`--async-episodes` and `--async-transitions` are OR thresholds; after a
threshold is reached, the trainer still waits until at least
`--min_batch_episodes` fresh episodes are available before updating. Defaults
are 32 episodes, 8,192 transitions, a lag of 4 updates, and a minimum of 128
episodes. Use `--max-policy-lag` and `--min_batch_episodes` to tune these
values. `--max-episode-steps` controls the existing simulator episode limit
and defaults to 1,800.

Each async update reports transition steps/second, active slot percentage,
episodes used and stale episodes dropped, mean episode length, mean and
median final distance, mean discounted return, mean policy lag, end-reason
counts, PPO clip fraction, approximate KL, and loss. Clip fraction and
approximate KL are averaged over the PPO minibatches and optimization epochs
in that update. The active percentage is the fraction of environment slots
stepped during collection ticks; the synchronous learner temporarily pauses
collection while its minibatch update runs.

Training risks to monitor:

- The policy is shared and updated between collection ticks; trajectories can
  contain actions from several policy versions. Any episode with even one
  transition beyond the lag limit is dropped in full, which can delay updates
  if the limit is too strict. The minimum batch threshold prevents tiny
  updates, but can make the trainer wait longer for enough fresh complete
  episodes.
- Monte Carlo returns are intentionally used only for completed episodes.
  Unfinished episodes and any finished buffer below its trigger threshold are
  discarded at shutdown, so use checkpoints/reports to detect interrupted
  runs.
- The episode cap changes the mix of terminal causes when set below 1,800.
  Treat "episode limit" as a training horizon, not a crash, when comparing
  returns.
- `train_batch` normalizes returns across transitions in the batch; long
  episodes therefore contribute more samples than short ones even though
  logging reports the mean return per episode.

The learning rate decays by `0.995` per learner update to a floor of `1e-5`.
`--updates`, `--report-every`, `--learning-rate-decay`,
`--optimization-epochs`, and `--seed` are also configurable. Legacy runs
resume from `car_ai_shared_checkpoint_v2_run_*.pt`; async runs default to the
separate `car_ai_async_checkpoint_v1_run_*.pt` family. Use `--checkpoint` to
select a different family explicitly.

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
