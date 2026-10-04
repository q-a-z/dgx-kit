# Release notes

Newest first. Every commit adds its entry here.

## 0.1.9 (in progress)

### Changed
- Version bump. Everything in 0.1.8 below is released; new changes are listed here from now on.

### Fixed
- **A context you set is kept as you wrote it.** The memory autotuner capped an explicit context at the config's `max_position_embeddings`, which would have lowered models that serve more than their config says (Nemotron 3.5 runs at 393K although its config says 256K). Only the default 32K context is now limited by the model's own maximum, so a small model isn't asked for more than it has.

- **A model that is up no longer stays on "Starting".** The model list dropped a model's live stats the moment its container wasn't "running", and that includes the second or two it is "created" right after Start (more likely when several models start at once). Nothing brought the stats back until the dashboard restarted, so a server that already answered still read "Starting". Only a crashed container is dropped now, and a running one that lost its stats is picked up again on the next refresh.

- **Every time is shown on a 24-hour clock.** Chart axes and hover readouts, the action log, benchmark and firmware dates and the update check no longer follow the browser's language: no more "3:07 PM" or "3pm", always "15:07".

### Added
- **Notes can be edited, and start from the real settings.** The model's Settings tab has a Notes box. When a model has no note yet it is filled with one line built from its saved settings ("Laguna-XS-2.1-Abliterated · vLLM nvfp4 · draft n=9, fp8 KV, ctx 262144"); edit it and press Save note. Fill from settings rebuilds that line after you change something, so it can't quietly disagree with the config. The box sits a little lower, clear of the settings above it.
- **The draft-token count shows next to the draft model** in the model panel ("draft Laguna-XS-2.1-DFlash · n=9"), read from the saved settings, so it is always the real number.
- **Model uptime.** A running model's state reads "Serving for 3 h 12 min" in its panel and in the model table, and hovering shows when it started. It counts from the last time its container started, so a restart resets it. Models DGX-kit didn't start show nothing, since Docker's start time isn't theirs to report. The service reports it as `container.started` in `GET /api/models`.
- **Tuning notes for a GB10** in the README: what speculative decoding and the Mamba SSM cache cost in KV cache, which flag combinations vLLM refuses, the parsers a Qwen model needs, and a pointer to recipes.vllm.ai, all with the numbers measured on a Spark.

## 0.1.8

### Added
- **"Why it stopped" for a model that crashed.** A model that stopped on its own now shows, in its panel, what its log says went wrong, in plain words, with a one-click fix saved as a new version of its settings (so Restore can undo it). Recognised: a KV cache too small for the context ("raise it to 7.2 GB" or "lower the context to 86,016"), other models holding too much memory, the GPU out of memory, an illegal memory access (offers to turn prefix caching off), an engine image that doesn't know the model, a missing file, a taken port, and settings the engine refuses together (for example `--enable-expert-parallel` with a speculative draft, or Mamba stochastic rounding without a float16 SSM cache). Anything else shows the last error line.
- **Benchmark history.** A model's Benchmark tab now keeps every finished run and shows them over time: a table (decode, prefill, concurrent streams, tool calls, needle, complex-code score), charts of decode and prefill speed, and a warning when the latest run is clearly worse than your usual (decode, prefill or total speed more than 8% below the median of the last five runs, latency 15% above, or a quality score 10 points down). Each run now records what the model ran on (engine image and its build, DGX-kit version, driver, context, KV cache type, draft settings), so the warning says what changed since the run before, for example "image: vllm-spark:0.29 → vllm-spark:0.30". Runs made before this have no such record, and show no changes.

### Fixed
- **Benchmark: a coding test that scored 0/9 now says why.** A solution that wouldn't load was reported as "JSONDecodeError" (the benchmark failing to read a result) instead of the real error, such as `IndentationError` or `NameError: name 'time' is not defined`. It now shows that error line. An answer cut off by the token limit (the model kept thinking aloud inside its code block until the limit) is now marked "cut off at the token limit, so this is not a fair score" and classed as `truncated`, not as a syntax error by the model.

### Changed
- Version bump. Everything in 0.1.7 below is released; new changes are listed here from now on.

## 0.1.7

### Added
- **Update DGX-kit from GitHub, from the page.** Settings → System → **DGX-kit update** shows the running version and the latest on GitHub, what's new, and an **Update** button: it downloads the latest source, builds the image, keeps the old one as `dgx-kit:previous` for a rollback, restarts the dashboard and reloads the page on the new version. Models and the gateway keep running, and your settings, keys and recipes are untouched. A yellow **Update 0.x.y** badge next to the logo appears when a newer version is out. It asks first, is off in read-only mode, and `DGXKIT_UPDATE_REPO` points it at another GitHub repository.
- **Update checking is a setting.** DGX-kit looks on GitHub for a newer version by itself once a day. Settings → System → DGX-kit update → **Check for updates** changes that to every hour, every week or never (then only **Check now** looks). The last look is remembered across restarts, so a restart doesn't cause a new one, and a failed look keeps the last good answer and says why it failed.
- Asking what is new no longer needs Docker to be running (it only needed it to start an update).

## 0.1.6

### Changed
- **The repository moved to https://github.com/q-a-z/dgx-kit**, and `dgx-kit update` fetches from there. Installs that still point at the old address keep working until you run an update from a package or a clone once (`dgx-kit update PATH`), which refreshes the command.
- Version bump. Everything in 0.1.5 below is released; new changes are listed here from now on.

## 0.1.5

### Fixed
- **Split GGUF models (`…-00001-of-00002.gguf`, `…-00002-of-00002.gguf`).** DGX-kit counted only one piece: the memory plan and "Check if it fits" were too small by the size of the other pieces, and each piece was offered as a separate model. A split GGUF is now one model: its pieces are added up for the plan, and only the first piece is offered (llama.cpp loads the others from it).
- **"start check failed … 503 Service Unavailable" after starting a llama.cpp model.** The ten-second speed check began as soon as llama.cpp answered its metrics, which it does while the model is still loading, and failed because completions are refused until loading ends. The check now waits for the model to report ready (up to five minutes) and starts its ten seconds then.

### Changed
- Version bump. Everything in 0.1.4 below is released; new changes are listed here from now on.

## 0.1.4

### Added
- **Memory autotuning: a model takes what its context needs, not all free memory.** Until now a model without a fixed KV cache was handed every free byte for it, so a 27B model could take 112 GB on a quiet machine. Now, unless you ask otherwise, the context defaults to 32K tokens (never more than the model supports) and the KV cache is sized for that context and the "min requests at once" setting, plus 5%. Settings show how much memory the model will use in all ("About 82 GB of memory in all"). **Use all free memory** (model settings) brings back the old behaviour for a dedicated box; models with a fixed KV cache size are not affected. New built-in templates: **compact** (8K context, one user), **long-context** (128K) and **all-memory**.
- **Models start without any image set-up.** Pressing Start on a model whose image isn't on the box now fetches it first (a pull, with progress) and starts the model by itself when it is there; the model shows "Fetching image", and Stop cancels it. A model that points at a local build nobody built (such as `dgx-kit/llamacpp:gb10`) just uses the engine's own image instead of asking for a build.
- **"Saved" notifications.** Saving a model's settings (including template changes), the gateway and LiteLLM settings, the Hugging Face token, the model folders, an engine's image, or the password now shows a short message in the corner of the screen.
- **Image pull progress.** Pulling an image shows a progress bar with the data downloaded ("3.2 GB of 8.4 GB downloaded · 5 of 12 layers"), in Settings, Engine images and in a model's settings. Choosing a tag the box doesn't have (Change tag, then Use this tag) now starts the pull straight away, so there is something to watch.
- **Pull an image from a model's settings, and watch it.** When the image picked for a model (or typed under "Other tag…") isn't on the box, its settings say so and offer **Pull**; while it downloads they show the layer progress ("12 of 30 layers"), and a failed pull shows why, with **Try again**. Before, only an engine's own image could be pulled, in Settings, Engine images.

### Fixed
- **GGUF models now load out of the box.** The default llama.cpp image is now `ghcr.io/ggml-org/llama.cpp:server-cuda13`, the CUDA 13 build for arm64 that runs on the GB10 (checked with a Gemma 4 GGUF), so nothing has to be compiled on the box. The GGUF file is found whether its name is written relative to the model folder or to the models folder (`gguf/model.gguf`).
- **A GGUF model whose file name doesn't match its folder** (for example `gguf/model.gguf` under a folder that already is `gguf/…`) reported "the model's config doesn't say how big its attention is". It now says which file wasn't found, and that the GGUF file name is relative to the model folder.
- **"Pull failed … repository does not exist" for dgx-kit/llamacpp:gb10 and the other GB10 images.** Those are built on the box from DGX-kit's own recipes and exist in no registry, yet the image picker offered to pull them. It now says "Not built on this box yet" with a **Build** button and shows the build's progress; an engine whose image is a local build builds it when you press Pull, and the pull request refuses such a tag with a clear message.
- **GGUF models (llama.cpp) failed to start with a 500 error, and "Check if it fits" did nothing.** A GGUF comes without a `config.json`, so DGX-kit had nothing to size the context and KV cache from. It now reads the model's layer count, attention heads and context length from the GGUF file's own header (including sliding-window and shared-KV layers, as in Gemma), for new and already-saved models alike. A model whose size can't be found is now refused with a plain message instead of an error page.
- **Copy buttons** (the LiteLLM key and address in Settings, Gateway, and the launch command of a model started elsewhere) did nothing when the dashboard was opened over plain http, because browsers only offer the clipboard on https. They now fall back to the older copy method, and say "Copy failed" if that fails too, instead of claiming "Copied".

### Changed
- Version bump. Everything in 0.1.3 below is released; new changes are listed here from now on.

## 0.1.3

### Changed
- The box's real model is part of the logo in the top bar (for example "ASUS GX10"), read from the firmware; hover it for the vendor, product family, GPU and BIOS. Where the firmware says nothing, the GPU name is shown instead.
- Version bump. Everything in 0.1.2 below is released; new changes are listed here from now on.

## 0.1.2

### Added
- **System tab** in Settings (the first tab): the machine's model, BIOS (version, date, vendor), GPU VBIOS, kernel, NVIDIA driver, CUDA, Docker and DGX-kit versions, and **firmware updates** from fwupd (the Linux firmware service the DGX Spark uses): every firmware device with its version and whether an update is on offer, a **Check for updates** button, and a check once a day. A **Changes** list remembers when any of these versions changed (a new BIOS, driver, kernel or firmware flash) and from what. DGX-kit only reports; installing is still `sudo fwupdmgr update` on the machine. The check runs in a short-lived container from DGX-kit's own image with access to the system bus; it is off in read-only mode.
- **Clock gauge** after GPU power: the GPU's clock (outer ring) and the CPU's average clock (inner ring), in MHz.
- **CPU cores gauge** beside the ring gauges: one short bar per core in a near-square grid (5 by 4 on the GB10), its caption level with the other gauges, the same view as the CPU tile at the bottom (amber from 85%, red from 97%; hover a bar for its core and clock).
- **Network and Disk gauges** next to Temperature, GPU power and Load: network in and out, disk read and write, in MB/s. The rings use a square-root scale, so light traffic still shows.
- **Speed arrows** on the Stats block: Decode and Prefill show a green ▲ when the speed now is at least the average, an amber ▶ when a little slower, a red ▼ when well below it, and nothing while idle.
- **LiteLLM indicator** in the top bar next to Live: green when the gateway answers and accepts the key, red when it doesn't (hover for why). It opens Settings, Gateway.
- **Traffic stats** on the home page next to the gauges, in six compact figures: tokens in and out, requests, decode and prefill speed now and on average, and requests running (with those waiting), summed over the running models. Totals count from when each model started.
- **Extra LiteLLM settings editor** in Settings, Gateway: edit `litellm/extra.env` in the page (Save, or Save and apply, which restarts the gateway for a few seconds), see which lines are ignored and whether the saved file is live. No root editing needed.

### Changed
- The GPU name (for example NVIDIA GB10) is shown next to the logo and version in the top bar, and the "System" heading is gone from the gauges card.
- The gauges and the stats line are centered in the System card.
- The bottom tiles that repeated the top gauges (GPU load, temperature, power, CPU, Disk, Network) are removed; the GPU clock and Unified memory tiles are gone too. **Hardware details** now holds what is on the GPU (and the Sensors tile, hidden until you add it with Arrange). Saved layouts keep working.
- The traffic figures sit in the System card in their own centered line under the gauges, behind a divider, without a caption. The gauges wrap onto a second line when the window is narrow.
- The "serving normally · free memory · gateway OK" line next to the logo is gone; the top bar shows a status there only when something needs attention.
- **CPU, disk, network** is always open at the bottom of the home page; the fold/unfold button in the System card is gone.
- The gateway address and the **Open LiteLLM** link moved from the home page to Settings, Gateway; the home page's System card no longer has a gateway block or the "Serves" list.
- The "Memory free" block is gone from the home page's System card (the memory map shows it).
- **API documentation:** [docs/API.md](docs/API.md) describes every route, sign-in, errors, read-only mode and the model fields; `/docs` on the dashboard is the generated reference.
- README now documents every file and folder in the settings and state folders, every `config.env` key, the other variables, the Docker objects and the `--user` equivalents.

## 0.1.1

### Added
- **Publish / Unpublish** buttons on each model: list it on the gateway or take it off without restarting it. Unpublish asks first.
- **Settings tabs:** Gateway, Hugging Face, Engine images, Model folders, Import, Delete from disk, Activity and Password are separate tabs, and the tab is in the address (`#/settings/images`).
- **`dgx-kit update`:** the installer leaves a command in `~/.local/bin`, so an update is one line: `dgx-kit update` fetches the latest from git, `dgx-kit update PATH` updates from a package (`.tar.gz`, `.tgz` or a GitHub `.zip`) or from a git clone or unpacked folder, and `dgx-kit version` shows what is running. An update needs no sudo password unless the service file changes, keeps the previous image for a rollback, and never touches your settings, keys, recipes or password. The installer options are `--updatepath`, `--updaterepo` and `--update`. Uninstalling removes the command too.
- **The running version** is shown next to the logo and on the sign-in page.
- **Bundled LiteLLM gateway** with its own Postgres, set up by one button (Settings, Gateway), with memory limits; the top link opens LiteLLM's admin page.
- **Hugging Face token** setting, checked when saved.
- **Benchmarks:** a Benchmark tab on each model runs `tools/bench.py` on request; a ten-second speed check runs after a model you started first answers.
- **A model's settings as one block of engine options**, the same lines an llmctl `.conf` holds.
- **llmctl import:** `$HOME` and `~` in a conf are your home, weights named for another machine's folders are found here by folder name, and an image the machine doesn't have is replaced by the GB10 build of the same vLLM release.
- **Deleting model files** only from Settings, Delete from disk, after four confirmations; removing a model never touches its files.
- **Confirmation dialogs** for Stop, Restart and Remove.
- **GB10 vLLM image builds** with the GB10 patches, FlashInfer 0.7.0 and the GB10 build settings.
- **Combined GPU/CPU temperature and load gauges**, with icons.
- **Installer:** `--user` (no sudo), unattended answers, `--uninstall`, `--dry-run`, a Dockerfile for the dashboard image, and `tools/make-dist.sh` for release packages.

- **Extra LiteLLM settings:** `litellm/extra.env` in the state folder takes your own LiteLLM environment variables, one `NAME=value` per line (for example `STORE_MODEL_IN_DB=True`). DGX-kit creates it with examples, applies it when the gateway is made again (the **Re-run LiteLLM setup** button), and the Gateway tab shows its path and what is applied. `LITELLM_MASTER_KEY` and `DATABASE_URL` stay managed by DGX-kit.

### Changed
- A re-run of the installer offers your earlier answers, keeps the admin password on Enter, and restarts the service on the new code.
- README rewritten for people, with screenshots.
- `.gitignore` now also keeps secrets (`.env` files, keys, certificates, a password file), logs and patch leftovers out of the repository, and ignores `state/` and `models/` only at the top of the tree, so a code folder with one of those names is never hidden. The image build skips the docs.

### Fixed
- Running DGX-kit from a clone, without the installer, no longer stops at start with a permission error on `/var/lib/dgx-kit`: the state folder then defaults to `~/.local/share/dgx-kit`, and the start-up line says where the state is kept. Set `DGXKIT_STATE_DIR` to choose another.
- The three-dot menu on a model now closes when you click elsewhere, press Escape, or choose an item (it used to stay open until you clicked the dots again).
- Stopping a model from the dashboard (or a clean stop from a shell) shows **Stopped**, not **Crashed**.
- A model that is still loading shows **Starting**, not "Not answering".
- Starting Nemotron no longer dies while compiling GPU kernels: containers share the kernel caches, the compile is capped at two jobs, and memory lock is lifted.
- Nemotron now gets a size plan (its config lists layers differently), and saving a model from the page no longer erases its stored config.
- The idle power-governor blip is no longer shown as GPU throttling.
- The installer's re-run was blocked by its own port and never restarted the service.
