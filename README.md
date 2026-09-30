# DGX-kit

A dashboard and model manager for the NVIDIA DGX Spark. It shows the machine's temperature, load and
memory; downloads, starts and stops language models; gives every running model one OpenAI-compatible
address through a built-in LiteLLM; and can benchmark a model on request.

You install one thing, the dashboard. It sets up everything else itself: LiteLLM, its database, the
engine images and the kernel caches.

## Screenshots

![The dashboard: GPU temperature, power and load, free memory, the LiteLLM gateway and the memory map](docs/screenshots/dashboard.png)

The dashboard shows the machine at a glance and every model as a block in unified memory.

![A model's settings as one block of engine options](docs/screenshots/model-settings.png)

A model's settings are one block of engine flags, the same lines an llmctl `.conf` holds. Delete a line to clear it.

![The confirmation dialog for deleting model files from disk, step 2 of 4](docs/screenshots/delete-confirm.png)

Deleting model files takes four confirmations and only works from Settings.

*The screenshots come from a demo instance with simulated GB10 readings, so they show no real machine.*

## What you need

- A DGX Spark running the factory **DGX OS** (it already has the NVIDIA driver, Docker and the NVIDIA container toolkit).
- About 100 GB free for models, and internet access on the first run.
- A user that can run `sudo`, or, for the no-sudo install below, one who is already in the `docker` group.

A plain Ubuntu machine is not enough: the installer does not install the NVIDIA driver or add NVIDIA's package repository.

## Install

1. Get the package onto the Spark and unpack it:

   ```
   tar xzf dgx-kit-0.1.1.tar.gz
   cd dgx-kit-0.1.1
   ```

2. Look before you leap (changes nothing):

   ```
   bash installer/install.sh --check
   ```

3. Install:

   ```
   bash installer/install.sh
   ```

   It asks a few questions; Enter keeps the default.

   | Question | What it means |
   |---|---|
   | Where should models be stored? | The folder for downloaded models. Default `~/models`. |
   | Dashboard port | Where you open the dashboard. Default 3000. |
   | Reachable from the LAN or this machine only | `lan` lets other computers on your network open it. |
   | Admin password | Protects the dashboard. At least 8 characters. Only a hash of it is kept. |
   | Hugging Face token | Optional. Needed only for gated or private models. You can add it later in Settings. |
   | Gateway port | Where the LiteLLM address is served. Default 4000. |
   | Pull engine images now | Downloads the engine images in the background (several GB). |
   | Update all system packages first | Default **no**. Saying yes can change the kernel or the NVIDIA driver. |

   The installer builds the dashboard image (about a minute), writes its settings to
   `/etc/dgx-kit`, and starts it as a system service that comes back after a reboot.

4. Open `http://<the Spark's address>:3000` and sign in.

Running the installer again is safe. It updates an existing install: unpack a newer package and run it.

### Updating

To update from a newer package, keep the `.tar.gz` (and its `.sha256` beside it, if you have one) and run:

```
bash installer/install.sh --updatepath /path/to/dgx-kit-0.1.1.tar.gz          # add --user for a user install
```

It checks the checksum, unpacks the package to a scratch folder, and updates from the packaged installer, so the newest
install logic always runs. Your settings, keys, recipes and admin password are kept, and it asks nothing. It rebuilds the
image, refreshes the service file and restarts the service; model containers keep running. The image it replaces is kept
as `dgx-kit:previous`; to go back, run `docker tag dgx-kit:previous dgx-kit:latest` and restart the service
(the command is printed at the end). `--dry-run` shows what it would do.

`bash installer/install.sh --update` does the same from a package you have already unpacked.
The first time, run the new package's installer (unpack it and use `--update`), since an older installer doesn't know these options.

### Without sudo

```
bash installer/install.sh --user
```

Everything is the same, but nothing needs administrator rights: the service is a user service, its
files live under your home folder, and the container runs as you, so the files it creates are yours.
It needs Docker and the NVIDIA container toolkit to be set up already, and you to be in the `docker`
group (log out and in after joining). If it can't set up something itself, it says so and stops.

To keep the service running after you log out, an administrator has to run `sudo loginctl enable-linger <you>` once;
the installer tries it and tells you if it couldn't.

### Unattended install

Every question can be answered ahead of time with an environment variable, so the installer can run
from a script. Names: `DGXKIT_INSTALL_` plus `MODELS_DIR`, `PORT`, `BIND`, `ADMIN_PASSWORD`, `HF_TOKEN`,
`GATEWAY_PORT`, `PULL_NOW`, `UPGRADE`. For example:

```
DGXKIT_INSTALL_ADMIN_PASSWORD='choose-one' DGXKIT_INSTALL_PORT=3000 bash installer/install.sh --user </dev/null
```

`DGXKIT_INSTALL_READONLY=yes` installs a watch-only dashboard that will not start, stop or pull anything.
`--dry-run` shows every command and the service file it would write.

## First steps in the dashboard

1. **Settings → Gateway.** LiteLLM starts by itself. If it isn't running, press **Set up LiteLLM**; it makes a key, pulls what it needs and starts LiteLLM with its database. The gateway key is printed at the end of the install and is kept in `/etc/dgx-kit/config.env` (`sudo grep LITELLM_MASTER_KEY /etc/dgx-kit/config.env`).
2. **Settings → Hugging Face.** Paste a token if you need gated models. It is checked when you save.
3. **Add a model.** Either add one from Hugging Face on the Models page, or, if you used llmctl, **Settings → Import from llmctl** turns your `.conf` files into models with the same folders, image and options. Nothing starts during an import.
4. **Start it.** Press Start. The model shows *Starting* while it loads (large models take minutes the first time, while engines compile kernels), then *Serving*. When it first answers, DGX-kit takes a ten-second speed check (decode and prefill) and shows it on the model.
5. **Use it.** Every running model that is marked to publish appears on the gateway address under its own name:

   ```
   curl http://<spark>:4000/v1/chat/completions \
     -H "Authorization: Bearer <gateway key>" -H "Content-Type: application/json" \
     -d '{"model": "<model name>", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

   **Open LiteLLM** at the top of the dashboard opens LiteLLM's own admin page (user `admin`, password is the gateway key).

## Everyday use

- **Settings of a model.** Its **Settings** tab is one text box of engine options, the same lines an llmctl `.conf` holds (`--max-model-len 393216`, `--moe-backend marlin`, and so on). Remove a line to clear it. If you don't fix the context or KV cache, DGX-kit sizes them from the memory that is free.
- **Stop and Restart** always ask first.
- **Benchmark.** A model's **Benchmark** tab runs the full battery or a quick run, only when you press the button. It refuses to run while other models are busy, because that would spoil the numbers.
- **Removing a model** (the three-dot menu) removes it from DGX-kit only. Its files stay on disk.
- **Deleting files from disk** is only possible in **Settings → Delete from disk**, after four confirmations, the last one typing the folder's name. A model that is running can't be deleted, and nothing outside your model folders can be.

## Where things live

| What | Where |
|---|---|
| Settings the installer wrote, including the gateway key | `/etc/dgx-kit/config.env` (`~/.config/dgx-kit/config.env` with `--user`) |
| Recipes and their version history, admin password hash, LiteLLM's database password, Hugging Face token (if set in Settings) | `/var/lib/dgx-kit` (`~/.local/share/dgx-kit` with `--user`) |
| LiteLLM's database | Docker volume `dgxkit-gateway-pg` |
| Downloaded models | the folder you chose (default `~/models`) |
| Compiled GPU kernels, shared by all models | `~/.cache/flashinfer` and `~/.cache/vllm-jit` |
| The service | `systemctl status dgx-kit` (`systemctl --user status dgx-kit` with `--user`) |
| Containers | `dgxkit-<model name>`, `dgxkit-gateway`, `dgxkit-gateway-db` |

Back up the settings folder and the state folder to keep your models' settings and keys.

## When something goes wrong

- **A model crashed.** Open its **Logs** tab; the line that says why is usually near the end. Stop clears the crashed container, and Start tries again.
- **Killed while compiling kernels** (`cicc` or `ninja` in the log, exit code 137). The container ran out of memory. DGX-kit already limits the compile to two jobs and keeps the kernels in `~/.cache`; if it still happens, raise the model's memory limit or start it when less is running.
- **"Free memory … is less than desired".** Another model holds memory. Stop something, or start the models one at a time.
- **The dashboard says a model isn't answering.** *Starting* is normal while it loads. *Not answering* means it answered before and stopped: check its Logs.
- **Port already in use.** Run the installer again and pick another port.
- **Can't sign in.** Run `bash installer/install.sh` again and set a new password.

## Uninstall

```
bash installer/install.sh --uninstall          # add --user if you installed with --user
```

It stops and removes the service, the dashboard container and its image, after you type `uninstall` to confirm.
Two more questions, both defaulting to no:

- **Remove the model containers and the LiteLLM gateway?** If you say no they keep running on their own.
- **Delete the saved settings, keys, recipes and LiteLLM's database?** There is no undo.

Your models and the compiled-kernel caches in `~/.cache` are never deleted. `--dry-run` shows every command first.
For a script, `DGXKIT_INSTALL_YES=yes`, `DGXKIT_INSTALL_REMOVE_MODELS` and `DGXKIT_INSTALL_REMOVE_DATA` answer the questions.

---

## For developers

```
pip install -e '.[dev]'
pytest
DGXKIT_FAKE_GPU=1 DGXKIT_STATE_DIR=./state DGXKIT_MODELS_DIR=./models \
  uvicorn --factory dgxkit.app:create_app --port 3000   # simulated GB10
curl localhost:3000/api/snapshot

cd web && npm install && npm run build   # page served at localhost:3000
npm run dev                              # or live reload on :5173, proxied to :3000
```

The web build needs Node 20.19 or newer. On a real Spark, drop `DGXKIT_FAKE_GPU`; everything the
service reads there is read-only. Build the release package from the last commit with
`bash tools/make-dist.sh`; it writes `dist/dgx-kit-<version>.tar.gz` and its checksum.

Layout:

- `dgxkit/collectors`: GPU (one NVML session, no `nvidia-smi`), CPU, memory, storage, network, sensors.
- `dgxkit/engines`: metrics parsers for vLLM, SGLang and llama.cpp.
- `dgxkit/control.py`, `dgxkit/api_models.py`: start, stop, edit, logs, downloads over HTTP. Only containers labelled `dgxkit.model` are touched.
- `dgxkit/options.py`, `dgxkit/importer.py`: a model's settings as text, and llmctl `.conf` import.
- `dgxkit/gateway.py`: the bundled LiteLLM and its Postgres.
- `dgxkit/benchmarks.py`, `tools/bench.py`, `dgxkit/quickcheck.py`: the on-request benchmark and the ten-second start check.
- `dgxkit/images.py`, `images/`: pinned engine images and local GB10 builds.
- `dgxkit/auth.py`: the admin password and sessions.
- `web/`: the dashboard (React, Vite, uPlot).
- `installer/install.sh`, `Dockerfile`: the install.

---

Vibecoded with love and Claude.
