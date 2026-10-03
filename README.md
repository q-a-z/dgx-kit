# DGX-kit

A dashboard and model manager for the NVIDIA DGX Spark. It shows the machine's temperature, load and
memory; downloads, starts and stops language models; gives every running model one OpenAI-compatible
address through a built-in LiteLLM; and can benchmark a model on request.

You install one thing, the dashboard. It sets up everything else itself: LiteLLM, its database, the
engine images and the kernel caches.

Release notes: [RELEASE_NOTES.md](RELEASE_NOTES.md).

## What it does

- **Watch the machine:** gauges for temperature, power, clock, load, network, disk and every CPU core, a stats line (tokens in and out, requests, decode and prefill speed against their averages), and the models laid out in unified memory. The logo carries the box's real model, read from its firmware (for example ASUS GX10).
- **Keep track of firmware:** Settings → System lists the BIOS, GPU VBIOS, kernel, driver and every firmware device fwupd manages, says when an update is available, and remembers when any version changed.
- **Run models:** start and stop them, edit their engine options as one block of text, and see speed and cache numbers live. A model takes the memory its context needs, not all that is free (it says how much), and Start fetches a missing engine image by itself. GGUF models run on llama.cpp with nothing to build.
- **One gateway:** every running model is published on one OpenAI-compatible address through a built-in LiteLLM, and you can publish or unpublish each one.
- **Bring your setup over:** import llmctl `.conf` files, including models whose weights are already on disk.
- **Measure:** a ten-second speed check after each start, and a full benchmark when you ask for it.
- **Build GB10 engine images:** vLLM with the GB10 patches and FlashInfer 0.7.0, from the Settings page.
- **Install, update, roll back and remove** with one script, with or without sudo; update from the page too, straight from GitHub.

## Screenshots

![The dashboard: gauges for temperature, power, clock, load, network, disk and CPU cores, the stats line and the memory map](docs/screenshots/dashboard.png)

The dashboard shows the machine at a glance, a stats line for the models' traffic, and every model as a block in unified memory.

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
   tar xzf dgx-kit-0.1.8.tar.gz
   cd dgx-kit-0.1.8
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

Running the installer again is safe: it offers your earlier answers as defaults, keeps your admin password if you press
Enter, and restarts the service on the new code. For updating, the next section is shorter.

### Updating

**From the page:** Settings → System → **DGX-kit update** compares the running version with the one on GitHub (looked up once a day by default; the **Check for updates** setting there makes it every hour, every week or never, and **Check now** always looks; a yellow **Update 0.x.y** badge appears next to the logo when a newer one exists), shows what's new, and **Update** does the rest after one confirmation: it downloads the latest source, builds the image, keeps the old image as `dgx-kit:previous`, and restarts the dashboard, which reloads by itself. Models and the gateway keep running. It is switched off in read-only mode, and `DGXKIT_UPDATE_REPO` changes the GitHub repository it uses.

**From a terminal:** the install leaves a `dgx-kit` command. Updating is one line:

```
dgx-kit update                      # fetch the latest from git and update
dgx-kit update ~/dgx-kit            # from a git clone (or any unpacked package folder)
dgx-kit update dgx-kit-0.1.8.tar.gz # from a package, a .tgz, or a GitHub .zip
dgx-kit version                     # what is running
```

An update needs no sudo password: it rebuilds the dashboard image, restarts the service, and keeps your settings, keys,
recipes and admin password. Model containers keep running. The image it replaces stays as `dgx-kit:previous`, and the
update prints the one command that goes back. `--dry-run` shows what it would do. It asks for sudo only if the service
file itself changed in that version.

- **The first time**, the command doesn't exist yet: unpack the new package (or use a clone) and run
  `bash installer/install.sh --update` once. That installs `dgx-kit` into `~/.local/bin` (add it to your `PATH` if the
  shell says *command not found*).
- **From git** needs this machine to be able to read the repository. For a private one, add a read-only deploy key and
  let `git` use it (an `~/.ssh/config` entry for the host). `DGXKIT_UPDATE_REPO=<url>` points the command at another repository.
- **From a file or a folder** needs no network. A checksum file `<name>.sha256` beside an archive is checked if it is there.
- **Add `--user`** to every installer command if you installed with `--user`; the `dgx-kit` command remembers that itself.

The same options exist on the installer itself: `--updatepath PATH` (archive or folder), `--updaterepo URL`, and `--update`
for the package you are standing in.

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

### All installer options

```
bash installer/install.sh [--user] [--check | --dry-run] [--uninstall | --update | --updatepath PATH | --updaterepo URL]
```

| Option | What it does |
|---|---|
| *(none)* | Check the machine, ask a few questions, install and start DGX-kit. Needs sudo. |
| `--user` | The same without sudo: a user service, files under your home. Combines with every other option. |
| `--check` | Read-only report of what is present and what would be installed. Changes nothing. |
| `--dry-run` | Print every command and the service file instead of running them. |
| `--updatepath PATH` | Update an existing install from a package on this machine: a `.tar.gz`, `.tgz` or `.zip`, or a folder (a git clone or an unpacked package). Keeps settings, keys, recipes and password; asks nothing. |
| `--updaterepo URL` | The same, fetching the latest from a git repository. |
| `--update` | The same from the package or clone you are standing in. |
| `--uninstall` | Remove the service and the dashboard; asks before touching anything else. |

Every question can be answered ahead of time with `DGXKIT_INSTALL_<NAME>` (see *Unattended install* and *Uninstall*).

## First steps in the dashboard

1. **Settings → Gateway.** LiteLLM starts by itself. If it isn't running, press **Set up LiteLLM**; it makes a key, pulls what it needs and starts LiteLLM with its database. The gateway key is printed at the end of the install and is kept in `/etc/dgx-kit/config.env` (`sudo grep LITELLM_MASTER_KEY /etc/dgx-kit/config.env`).
2. **Settings → Hugging Face.** Paste a token if you need gated models. It is checked when you save.
3. **Add a model.** Either add one from Hugging Face on the Models page, or, if you used llmctl, **Settings → Import** turns your `.conf` files into models with the same folders, image and options. Nothing starts during an import. `$HOME` and `~` in a conf mean your home, and weights named for another machine's folders are found here by folder name. If a conf names an image this machine doesn't have, the import uses the GB10 build of the same vLLM release; build it first under **Settings → Engine images**.
4. **Start it.** Press Start. The model shows *Starting* while it loads (large models take minutes the first time, while engines compile kernels), then *Serving*. When it first answers, DGX-kit takes a ten-second speed check (decode and prefill) and shows it on the model.
5. **Use it.** Every running model that is marked to publish appears on the gateway address under its own name:

   ```
   curl http://<spark>:4000/v1/chat/completions \
     -H "Authorization: Bearer <gateway key>" -H "Content-Type: application/json" \
     -d '{"model": "<model name>", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

   **Open LiteLLM** at the top of the dashboard opens LiteLLM's own admin page (user `admin`, password is the gateway key).

## Everyday use

**Each model has a panel** with four tabs: **Live** (speed, queue, KV cache, draft acceptance), **Settings**, **Logs** and **Benchmark**.

- **Publish / Unpublish.** The button in a model's panel lists it on the gateway or takes it off, right away, without restarting the model. Unpublish asks first, because clients lose access. The panel says whether the model is *published*. A stopped model publishes as soon as it runs.
- **Settings of a model.** One text box of engine options, the same lines an llmctl `.conf` holds (`--max-model-len 393216`, `--moe-backend marlin`, and so on). Remove a line to clear it. Unless you fix the context or the KV cache, DGX-kit picks a 32K context (never more than the model supports) and sizes the KV cache for it and for the requests at once, so a model takes what it needs and not all the free memory. The settings show the total ("about 82 GB in all"); **Use all free memory** gives the longest context that fits to a box that runs one model. The templates **compact**, **balanced**, **long-context**, **many-users** and **all-memory** set these in one click.
- **Images are fetched for you.** Press Start on a model whose engine image isn't on the box and DGX-kit pulls it, with a progress bar, and starts the model by itself when it is there (Stop cancels). llama.cpp models use the upstream CUDA 13 image, which runs on the GB10 as it is. Images built from DGX-kit's own recipes (the GB10 vLLM images) are optional extras; a model that names one nobody built uses the engine's own image.
- **"Saved" messages.** Saving settings, templates, the gateway, the Hugging Face token, folders, an image or the password shows a short message in the corner.
- **Start, Stop and Restart.** Stop and Restart always ask first. A model you stopped shows **Stopped**; one that died shows **Crashed** with its exit code.
- **Speed check.** When a model you started first answers, DGX-kit takes a ten-second check (two short decodes, two prefills) and shows it on the model. It is skipped when other models are busy.
- **Benchmark.** The **Benchmark** tab runs the full battery or a quick run, only when you press the button. Finished runs are kept: the **History** at the top of the tab charts decode and prefill speed over time, lists every run with what changed since the one before (engine image, DGX-kit, driver, settings), and warns when the latest run is clearly worse than your usual. It refuses to run while other models are busy, because that would spoil the numbers.
- **Removing a model** (the three-dot menu) removes it from DGX-kit only. Its files stay on disk.
- **Deleting files from disk** is only possible in **Settings → Delete from disk**, after four confirmations, the last one typing the folder's name. A model that is running can't be deleted, and nothing outside your model folders can be.
- **Version.** The running version is shown next to the logo, and on the sign-in page.

## Settings

Settings has one tab per topic (**System** is the first), and the tab is in the address (`#/settings/images`), so a link opens the same one.

| Tab | What it is for |
|---|---|
| System | **DGX-kit update** from GitHub (check, what's new, one-button update); the box's model, BIOS, GPU VBIOS, kernel, driver, CUDA, Docker and DGX-kit versions; firmware devices and updates from fwupd (a check once a day, or **Check for updates**); and a list of when any version changed. It only reports: install updates with `sudo fwupdmgr update`. |
| Gateway | The LiteLLM address and key, the **Set up LiteLLM** button, what the gateway serves, and an editor for its extra LiteLLM settings. |
| Hugging Face | A token for gated or private models. Checked when you save. |
| Engine images | The Docker images models run in; pull, change a tag (a tag the box doesn't have is pulled at once, with a progress bar), or build the optional GB10 vLLM images (patches, FlashInfer 0.7.0). |
| Model folders | Where DGX-kit looks for models. |
| Import | Turn llmctl `.conf` files into models. |
| Delete from disk | The only place model files can be deleted (four confirmations). |
| Activity | What DGX-kit has done on this machine. |
| Password | Change the admin password (shown when a password is set). |

## Where things live

DGX-kit keeps three kinds of things apart: **settings** the installer wrote (one file), **state** it creates while running (a folder), and **your data** (models and caches). A system install (with sudo) and a `--user` install use different folders:

| | System install | `--user` install |
|---|---|---|
| Settings folder | `/etc/dgx-kit` | `~/.config/dgx-kit` |
| State folder | `/var/lib/dgx-kit` | `~/.local/share/dgx-kit` |
| Service | `systemctl status dgx-kit` | `systemctl --user status dgx-kit` |

Both are owned by root on a system install, so reading or editing them by hand needs `sudo`. When you run from a clone without the installer, the state folder is `/var/lib/dgx-kit` if you can write there, else `~/.local/share/dgx-kit`; set `DGXKIT_STATE_DIR` to choose. The start-up line prints the folder ("DGX-kit state folder: …"). The state folder is never part of the repository.

### The settings file: `config.env`

`/etc/dgx-kit/config.env` (mode 600, root only): one `NAME=value` per line, read by the service and passed to the container. The installer writes it from your answers.

| Key | Meaning | Default |
|---|---|---|
| `DGXKIT_MODELS_DIR` | The folder models are downloaded to and looked for in | `~/models` |
| `DGXKIT_PORT` | The dashboard port | `3000` |
| `DGXKIT_BIND` | The address the dashboard listens on (`0.0.0.0` = the network, `127.0.0.1` = this machine only) | `0.0.0.0` |
| `DGXKIT_GATEWAY_PORT` | The LiteLLM gateway port | `4000` |
| `LITELLM_MASTER_KEY` | The gateway key. Also shown and copyable in Settings, Gateway | generated |
| `DGXKIT_PULL_IMAGES` | `yes` pulls the engine images at start | `no` |
| `DGXKIT_CACHE_DIR` | Where the shared compiled-kernel caches (`flashinfer`, `vllm-jit`) live | `~/.cache` |
| `DGXKIT_HOME` | The real home folder, so `~` and `$HOME` in an imported llmctl conf mean your home | your home |
| `HF_TOKEN` | A Hugging Face token (can also be set in Settings) | empty |
| `DGXKIT_READONLY` | `1` hides everything that changes things: a look-only dashboard | off |

To change one: edit the file with `sudo`, then restart (`sudo systemctl restart dgx-kit`) or re-run the installer, which offers your earlier answers. The admin password is **not** in this file; only its hash is stored, in `admin.pw` (below).

Other variables the program reads, mostly for developers: `DGXKIT_STATE_DIR` (state folder), `DGXKIT_MODEL_PATHS` (extra model folders, colon separated), `DGXKIT_WEB_DIR` (the built web page), `DGXKIT_GPU_INDEX` (which GPU to show), `DGXKIT_NO_PASSWORD` (run without a password), `DGXKIT_FAKE_GPU` and `DGXKIT_ROOT` (a demo instance without a GPU), `DGXKIT_IMAGE_VLLM`, `DGXKIT_IMAGE_SGLANG`, `DGXKIT_IMAGE_LLAMACPP`, `DGXKIT_IMAGE_LITELLM`, `DGXKIT_IMAGE_POSTGRES` (engine image tags), `DGXKIT_FLASHINFER` (the FlashInfer version the GB10 vLLM builds install, `0.7.0`), `DGXKIT_VLLM_029_BASE` and `DGXKIT_VLLM_030_BASE` (the vLLM releases those builds start from), and for the installer `DGXKIT_INSTALL_*` and `DGXKIT_UPDATE_REPO` (see the installer options).

### The state folder

| File or folder | What it holds |
|---|---|
| `admin.pw` | A scrypt hash of the admin password (never the password itself) |
| `session.key` | The key that signs login cookies. Deleting it logs everyone out |
| `settings.yaml` | What you set in Settings: the gateway address, Model folders |
| `gateway.key` | A gateway key entered in Settings, if you set one there |
| `hf.token` | The Hugging Face token entered in Settings (mode 600) |
| `images.yaml` | Your engine image choices (Settings, Engine images) |
| `models/` | One `<name>.yaml` recipe per model: the engine, its options, and the gateway and sizing settings |
| `models-history/` | Every earlier version of each recipe, for the version diff and restore |
| `templates/` | Recipe templates |
| `quick/` | The latest ten-second speed check of each model |
| `bench/` | Benchmark runs and their results |
| `user-stopped.json` | Which models you stopped yourself, so a stop is shown as **Stopped** and not **Crashed** |
| `litellm/config.yaml` | The LiteLLM config DGX-kit **generates**. It is rewritten on every gateway setup: don't edit it |
| `litellm/db.password` | The password of the gateway's Postgres |
| `litellm/master.key` | The gateway key, only when none is set in `config.env` or Settings |
| `litellm/extra.env` | Your own LiteLLM settings. Edit it in Settings, Gateway (below) |

### Extra LiteLLM settings

LiteLLM reads its options from environment variables. Anything DGX-kit doesn't set itself goes in `litellm/extra.env`, one `NAME=value` per line, `#` for comments. A new install creates it with commented examples (`# STORE_MODEL_IN_DB=True`, `# LITELLM_LOG=INFO`).

The easy way: **Settings, Gateway, Extra LiteLLM settings**. Edit the text, press **Save**, then **Save and apply**, which makes the gateway again (it restarts for a few seconds; models keep running). Lines DGX-kit can't use are listed under the box. The gateway is recreated whenever the file's content changes, and **Re-run LiteLLM setup** does the same. `LITELLM_MASTER_KEY` and `DATABASE_URL` are managed by DGX-kit and ignored here. To enable LiteLLM's model database, add `STORE_MODEL_IN_DB=True`.

DGX-kit itself sets, for the gateway: `LITELLM_MASTER_KEY`, `DATABASE_URL`, `NUM_WORKERS=1`, `LITELLM_LOG=ERROR`, `LITELLM_DISABLE_NO_REDIS_WARNING=true`, and a 4 GB memory limit.

### Docker objects

| What | Name |
|---|---|
| The dashboard | container and image `dgx-kit` (the previous image is kept as `dgx-kit:previous`) |
| One per running model | container `dgxkit-<model name>` |
| The gateway and its database | containers `dgxkit-gateway` and `dgxkit-gateway-db` (Postgres 16, reachable only on this machine, port 5433) |
| LiteLLM's database | Docker volume `dgxkit-gateway-pg` |
| Images DGX-kit builds | `dgx-kit/vllm:<series>-gb10` |

### Files outside those folders

| What | Where |
|---|---|
| The `dgx-kit` command and the installer copy it runs | `~/.local/bin/dgx-kit`, `~/.local/share/dgx-kit-installer` |
| The service unit | `/etc/systemd/system/dgx-kit.service` (`~/.config/systemd/user/dgx-kit.service` with `--user`) |
| Downloaded models | the folder you chose (default `~/models`) |
| Compiled GPU kernels, shared by all models | `~/.cache/flashinfer` and `~/.cache/vllm-jit` |

Back up the settings folder and the state folder to keep your models' settings and keys; for the gateway's own data (users, keys it issued, stored models) also back up the `dgxkit-gateway-pg` volume. Uninstalling never deletes models or caches.

## API

Everything in the dashboard is available over HTTP, for scripts. See [docs/API.md](docs/API.md); a live reference is at `/docs` on the dashboard.

## When something goes wrong

- **A model crashed.** Open its **Logs** tab; the line that says why is usually near the end. Stop clears the crashed container, and Start tries again.
- **A model you stopped shows Crashed.** DGX-kit calls a stop clean when it made the stop itself, when the engine logged "Application shutdown complete", or when the exit code is 0 or 143. A stop that Docker has to force-kill (exit 137) from outside the dashboard can still look like a crash.
- **"Weights not found".** A model imported with a folder path has no files there. Put them at that path, or add the folder that has them under **Settings → Model folders** and import again.
- **The gateway says "Key rejected".** The key saved in DGX-kit isn't the one LiteLLM runs with. Under **Settings → Gateway**, save the right key, or press **Re-run LiteLLM setup**.
- **Killed while compiling kernels** (`cicc` or `ninja` in the log, exit code 137). The container ran out of memory. DGX-kit already limits the compile to two jobs and keeps the kernels in `~/.cache`; if it still happens, raise the model's memory limit or start it when less is running.
- **"Free memory … is less than desired".** Another model holds memory. Stop something, or start the models one at a time.
- **The dashboard says a model isn't answering.** *Starting* is normal while it loads. *Not answering* means it answered before and stopped: check its Logs.
- **Port already in use.** Run the installer again and pick another port. It shows what is listening on the one you asked for, and a re-run accepts the port its own running service already holds.
- **An update went wrong.** Roll back with `docker tag dgx-kit:previous dgx-kit:latest` and `docker stop dgx-kit` (the service brings it back on the old image); the update prints the exact command.
- **`dgx-kit: command not found`.** Add `~/.local/bin` to your `PATH`, or run `bash installer/install.sh --update` from a package or clone to install the command.
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
