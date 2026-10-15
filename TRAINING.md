# Training and running

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
<<<<<<< HEAD
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

## Observation and traffic A/B options

By default, training retains the original 21-value observation and two
256-unit tanh layers. `--obs_v2` / `--obs-v2` selects 26 ego-relative values:
four traffic slots (ahead in lane, ahead-left, ahead-right, nearest behind),
each with proximity-weighted relative distance, lateral offset, relative
speed, TTC, and presence; then lane offset, road heading error, curvature at
10/20/40 m, and own speed. Empty traffic slots use distance 1 and presence 0.
Heading is estimated from the road centerline because the simulator does not
track the vehicle's yaw separately. Enabling this mode also randomizes initial
speed, lateral position, and traffic spawn positions.

`--hidden` sets comma-separated tanh layer widths, for example
`--hidden 64,64`; its default is `256,256`. New observation/network variants
use orthogonal initialization for hidden layers and a small output-layer gain
to start with low-magnitude action logits. This is a categorical policy, so
there is no continuous-action standard deviation; the existing entropy bonus
(0.01) encourages early action exploration.

Traffic controls are `--traffic_density` (active-car cap, default 14, zero
allowed), `--traffic_speed` (km/h, default 20), and `--ai_min_speed` (km/h,
default 40). `--curriculum` uses a rolling mean of the last 100 completed
episode distances to raise the traffic cap by one car per 100 m of mean
distance; it begins with 0-1 cars spawned within the first 100 m. It is
bounded by `--traffic_density`.

Example A/B commands (run from the repository root):

```powershell
# (a) Existing observation/network/training setup
.\quantum\Scripts\python.exe .\car.py --mode headless --async_reset --environments 1000 --updates 100 --checkpoint .\baseline_old.pt

# (b) New observations and smaller network, no traffic
.\quantum\Scripts\python.exe .\car.py --mode headless --async_reset --obs_v2 --hidden 64,64 --traffic_density 0 --environments 1000 --updates 100 --checkpoint .\obs_v2_no_traffic.pt

# (c) New observations/network with traffic curriculum
.\quantum\Scripts\python.exe .\car.py --mode headless --async_reset --obs_v2 --hidden 64,64 --traffic_density 14 --traffic_speed 20 --ai_min_speed 40 --curriculum --environments 1000 --updates 100 --checkpoint .\obs_v2_curriculum.pt
```

Model variants automatically use separate default checkpoint names based on
observation version, hidden widths, and collector mode. Reusing a custom
`--checkpoint` across incompatible observation sizes or network shapes will
fail to load; use separate checkpoint paths for A/B runs. GUI mode currently
supports only the original observation and network options.
=======
.\quantum\Scripts\python.exe .\car.py --mode headless --device cpu --environments 32 --until-stopped
```

Continuous training runs until `Ctrl+C`. A checkpoint is written after each
completed update, so stopping does not discard completed work.

The learning rate decays by `0.995` per shared update to a floor of `1e-5`.
`--learning-rate-decay`, `--optimization-epochs`, `--report-every`, and
`--seed` can also be adjusted. Training resumes automatically from
`car_ai_shared_checkpoint_v2_run_*.pt`; use `--checkpoint` to select another
checkpoint family. Model checkpoints are local artifacts and are ignored by
Git. The GUI selects the most recently saved compatible checkpoint from the
GUI and shared-policy training families, then displays its source and progress
in the status bar.
>>>>>>> 3530412f0987bce6e0e4e8e6cfe4f2121a1ee46a

Verify CUDA in the configured environment with:

```powershell
.\quantum\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

For Python 3.12 on Windows, a CUDA-enabled PyTorch build can be installed from
the official wheel index. Choose a wheel compatible with the installed NVIDIA
driver and CUDA toolkit:

```powershell
.\quantum\Scripts\python.exe -m pip install --upgrade "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

If CUDA is unavailable, run with `--device cpu` or install a compatible
CUDA-enabled PyTorch build and rerun the verification command.
