# DigitalOcean Benchmark Guide

These instructions explain how to run the MuJoCo render benchmark on a manually provisioned DigitalOcean droplet. The steps assume you have **already created the droplet yourself**; the repository never attempts to allocate cloud resources automatically.

## 1. Choose a Droplet

- **OS**: Ubuntu 22.04 LTS (recommended).
- **Instance type**: A CPU-optimized droplet (e.g., 4 vCPUs, 8 GB RAM) works well for the MuJoCo CPU renderer used in headless environments. Dedicated CPUs improve repeatability.
- **Storage**: 40 GB disk is typically enough for the repo, dependencies, and generated output.
- **Networking**: Ensure SSH access is enabled and note the public IP.

> **GPU rendering**: DigitalOcean GPU droplets are limited and come with additional setup requirements. The current automation focuses on the CPU (`--backend cpu`) path so it can run on any standard droplet.

## 2. One-Time Droplet Preparation

Log into the droplet (replace `root@<ip>` with your own username/IP address):

```bash
ssh root@<ip-address>
```

Install Git and clone the repository:

```bash
apt-get update -y
apt-get install -y git
git clone https://github.com/<your-org>/mojo_codex.git
cd mojo_codex
```

Run the bootstrap script to install system packages, Pixi, and the project’s Python dependencies:

```bash
scripts/do/bootstrap.sh
```

This installs:

- Required Ubuntu libraries for MuJoCo’s CPU/OSMesa backend (`libosmesa6`, `libgl1`, etc.).
- Pixi (the project’s package/environment manager).
- The benchmark’s Python requirements via `pixi install`.

If you open a new shell later, remember to add Pixi to your `PATH`:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

## 3. Run the Benchmark

The repository ships with a helper that executes a CPU-friendly run and saves frames:

```bash
scripts/do/run_benchmark.sh outputs/do_run_$(date +%Y%m%d_%H%M%S)
```

The command above:

- Uses the CPU backend (`--backend cpu`) to avoid relying on GPU drivers.
- Renders the default `workbench` scene at 640×480 for 5 seconds.
- Writes artifacts to `outputs/do_run_<timestamp>/`.

Feel free to run customized commands instead. Example (longer run, no frames):

```bash
pixi run python -m src.benchmarks.render_bench \
  --backend cpu \
  --scene apartment \
  --duration 10 \
  --width 800 \
  --height 600 \
  --output outputs/apartment_cpu \
  --target-fps 60 \
  --save-frames
```

To compare backends explicitly (e.g., CPU only on a droplet), remember to repeat `--backend` for each one you want to include:

```bash
pixi run python -m src.benchmarks.render_bench compare \
  --backend cpu \
  --duration 0.5 \
  --warmup-frames 0 \
  --target-fps 30 \
  --width 256 \
  --height 256
```

## 4. Collect Results

Download outputs back to your workstation (from your local machine):

```bash
scp -r root@<ip-address>:~/mojo_codex/outputs/do_run_* ./outputs_from_do/
```

The metrics JSON can be parsed or aggregated locally. Frames (if saved) are stored under `frames/` inside each output directory.

## 5. Cleanup

Once you have copied the necessary data:

- Remove the droplet in the DigitalOcean dashboard to avoid ongoing charges.
- Alternatively, keep the droplet running, but remember to shut it down when idle.

## Optional: Automate via Pixi Tasks

If you prefer running through Pixi tasks instead of shell scripts, the following command is equivalent to `scripts/do/run_benchmark.sh`:

```bash
pixi run python -m src.benchmarks.render_bench --backend cpu --scene workbench \
  --duration 5 --width 640 --height 480 --output outputs/do_run --save-frames
```

You can adjust arguments to match your experiment.

## Troubleshooting

- **`MUJOCO_GL` errors**: Ensure you are using `--backend cpu`. The bootstrap script installs OSMesa, which MuJoCo uses automatically in CPU mode.
- **Pixi not found**: Add `~/.pixi/bin` to your `PATH` (see above) or restart the shell after bootstrap.
- **Large output directories**: Clean up `/outputs` after downloading artifacts.

With these steps, you can spin up droplets on demand, run benchmarks manually, and keep tight control over cloud spend.
