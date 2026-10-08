# Release notes

Newest first. Every commit adds its entry here.

## 0.1.22

### Fixed
- **Safari showed no tab icon.** Safari doesn't take an SVG tab icon, so the page now also offers a 32 px PNG, a `favicon.ico` (which Safari asks for by default) and an `apple-touch-icon.png` for the home screen and bookmarks. Safari keeps tab icons in its own cache; if it still shows none, quit and reopen it once after the update.

## 0.1.21

### Fixed
- **The tab icon never showed.** The server answered every address that wasn't a known route, `/favicon.svg` included, with the dashboard page itself, so browsers got HTML instead of an icon. Real files in the page folder are now served as themselves. If your tab still shows the old icon after the update, reload with the cache cleared (browsers hold on to tab icons).

## 0.1.20

### Changed
- **The logo is in the dashboard's sand colors**, in the header, on the sign-in page and as the browser tab icon (`web/public/favicon.svg`), instead of green.

## 0.1.19

### Added
- **A logo.** A grid of GPU cores with a lit signal path through it, in NVIDIA green on black. It is the browser tab icon (replacing the placeholder one), sits beside the name in the header and tops the sign-in page. One drawing in two places: `web/public/favicon.svg` and `web/src/Logo.tsx`.

## 0.1.18

### Added
- **A "50% cached" column beside "90% cached" in the "$$$" table.** The shares are `CACHE_SHARES` in `web/src/pricing.ts`.

## 0.1.17

### Added
- **A "90% cached" column in the "$$$" table.** It prices the same tokens with 90% of the input read from the provider's cache at its cache-read price (`cached` in `web/src/pricing.ts`), the case for agents that resend a long shared prefix. The share is `CACHE_SHARE` in the same file; the other columns still pay full price for every input token.

## 0.1.16

### Fixed
- **The "$$$" table broke large numbers across lines.** The window is wider now, amounts stay on one line, the price sits under the model name, and billions of tokens show as "B".

## 0.1.15

### Added
- **A "$$$" button in the top menu.** It opens a table of what the tokens served since the models started would have cost at Claude (Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 4.5) and OpenAI (GPT-5.5, GPT-5, GPT-5 nano) list prices, input and output separately. Prices are in `web/src/pricing.ts` with the date they were taken; caching and batch discounts are not counted.

## 0.1.14

### Changed
- Version bump. Everything in 0.1.13 below is released; new changes are listed here from now on.

### Fixed
- **The download speed was wrong and jumped about.** It was the change since the previous time anything asked for the progress, and the downloads list, a model's status and every open tab each ask on their own schedule, so it often measured a fraction of a second and swung between almost nothing and far too much (the time left with it). It is now measured over the last ten seconds, whoever asks and however often.

## 0.1.13

### Changed
- Version bump. Everything in 0.1.12 below is released; new changes are listed here from now on.

### Fixed
- **The Download button of a decision model (Laya, Lev, Bekko) answered "not found".** It called the route for ordinary models instead of the service's own. It now starts the download, and shows its progress.

## 0.1.12

### Changed
- Version bump. Everything in 0.1.11 below is released; new changes are listed here from now on.

### Added
- **A Download button for each decision model.** Laya, Lev and Bekko each get their own: one press fetches everything that model needs and sets it up, and starts nothing. It downloads the model's repos in order (for Lev, the adapter and then the Qwen3.5-4B backbone the adapter names; for Bekko, without its 1.4 GB browser model), shows the progress with pause and cancel like any model download, then builds the model's image so that Start is immediate. A model that isn't downloaded shows **Not downloaded** with a download arrow beside it in the map and the list and a Download button in its panel; a failed download says why and Download continues from where it stopped. The service API has `POST /api/services/{name}/download`, and a service's status carries `download` and `setup`.
- The downloader can fill a Hugging Face cache folder and skip files, which Lev and Bekko need.

### Fixed
- A Laya folder that is still downloading, or has a config without its weights, is no longer taken for a ready checkpoint.

## 0.1.11

### Changed
- Version bump. Everything in 0.1.10 below is released; new changes are listed here from now on.

### Added
- **Lev and Bekko, two more decision models, set up beside Laya (not started).** Both answer the same typed questions as Laya on the same `/v1/systemone`, so a client switches by changing the address. They appear in the memory map and the model list like Laya, with the same panel (Live with Test, Settings, Logs) and Start, Stop and Restart. **Lev** (`interfaze-ai/lev`, a LoRA adapter on Qwen3.5-4B) wants about 11 GiB, so it is meant as a spare started on purpose: Start is refused while that much memory isn't truly free (the message says to drop the page cache or stop something), it stops with a message instead of crawling on the CPU if the GPU can't be opened, and its server has no API key, so it listens on this machine only until you tick **Listen on the network**. **Bekko** (`hotchpotch/bekko-system-one-v0-400m`, a 395M-parameter English encoder) wants about 3 GiB and asks for a key like Laya; it ships no server, so the image carries a small one that speaks Laya's request and answer shapes (choice, yes/no and score questions, batches, `min_confidence`). Their files live in hidden Hugging Face caches in the models folder (`.lev`, `.bekko`) so the Qwen3.5-4B backbone isn't listed under **Models on disk**; the panel says what is missing. Both run in one image, `dgx-kit/decision:gb10`, built on the first Start on the vLLM image DGX-kit already uses (it adds only scikit-learn, peft, sentence-transformers and Lev's source; a few minutes, longer when package downloads are slow).
- The service API takes the service's name (`/api/services/{name}/...`) and reports for each what can be picked (`choices`), whether it has a key, whether it can listen on the network and how much memory it wants free.
- **Start can be refused for lack of memory**, with the reason, for services that declare what they need.

## 0.1.10

### Changed
- Version bump. Everything in 0.1.9 below is released; new changes are listed here from now on.
- **The dashboard is a little wider** on large screens: the page and its header have 30 px more room on each side (the page is 1380 px wide at most, up from 1320).

### Added
- **Laya, a decision model, listed with the models.** Laya answers typed questions about a text (yes/no, a choice, a score) with calibrated probabilities in one pass, about 50 ms on the GB10: guardrails, routing, triage. It isn't a chat model, so it can't run on vLLM or sit behind the gateway; its own HTTP server (`POST /v1/systemone`) runs in a container that DGX-kit builds (`dgx-kit/laya:gb10`, on the vLLM image it already uses), starts, stops and watches. It appears in the memory map and the model list like any model, with its GPU memory, its state ("Serving for 3 h 12 min") and Restart, Stop and Logs. Its panel has **Live** (a **Test** button that asks it a sample question and times the answer), **Settings** and **Logs**, and copies its address and its API key (it asks for one). A missing or outdated image is built first and the server starts by itself afterwards.
- **Pick which Laya checkpoints run.** The Settings tab lists English, multilingual and typed-decisions as found in the models folder (the ones it doesn't have are greyed out) and runs only the ones you tick; it says when the running set differs from the pick and a restart would change it. The device (GPU or CPU) and the port are there too.
- **A request never goes to a checkpoint that isn't running.** Without a `model`, Laya picks English or multilingual by the text's language; if you run only typed-decisions that sent the request to a checkpoint that isn't loaded and the server tried to download it (a 500 and a long traceback in the log). Now the first checkpoint that runs answers, and asking for one that isn't running is refused with a plain 422 that lists what is.
- **On the GB10 the GPU can be unavailable for a while:** CUDA needs memory that is truly free, and the page cache (a model or an image just read) can hold nearly all of it while Linux still reports gigabytes as "available". Laya's container now starts on the CPU in that case and says so (about 1 s a question instead of 50 ms; restart it after freeing memory to get the GPU back). The first GPU call in the container is also a tiny one, because when it is the weight copy itself the GB10 reports "out of memory" and Laya silently falls back to the CPU.
- Laya's checkpoint folders no longer show up under **Models on disk** as if an engine could run them.
- **Notes can be edited, and start from the real settings.** The model's Settings tab has a Notes box. When a model has no note yet it is filled with one line built from its saved settings ("Laguna-XS-2.1-Abliterated · vLLM nvfp4 · draft n=9, fp8 KV, ctx 262144"); edit it and press Save note. Fill from settings rebuilds that line after you change something, so it can't quietly disagree with the config. The box sits a little lower, clear of the settings above it.
- **The draft-token count shows next to the draft model** in the model panel ("draft Laguna-XS-2.1-DFlash · n=9"), read from the saved settings, so it is always the real number.
- **Model uptime.** A running model's state reads "Serving for 3 h 12 min" in its panel and in the model table, and hovering shows when it started. It counts from the last time its container started, so a restart resets it. Models DGX-kit didn't start show nothing, since Docker's start time isn't theirs to report. The service reports it as `container.started` in `GET /api/models`.

### Fixed
- Comments in the source and the vLLM patch notes no longer name a specific machine, user or client application.
- **A model that is up no longer stays on "Starting".** The model list dropped a model's live stats the moment its container wasn't "running", and that includes the second or two it is "created" right after Start (more likely when several models start at once). Nothing brought the stats back until the dashboard restarted, so a server that already answered still read "Starting". Only a crashed container is dropped now, and a running one that lost its stats is picked up again on the next refresh.
- **Every time is shown on a 24-hour clock.** Chart axes and hover readouts, the action log, benchmark and firmware dates and the update check no longer follow the browser's language: no more "3:07 PM" or "3pm", always "15:07".

## 0.1.9

### Changed
- Version bump. Everything in 0.1.8 below is released; new changes are listed here from now on.

### Fixed
- **A context you set is kept as you wrote it.** The memory autotuner capped an explicit context at the config's `max_position_embeddings`, which would have lowered models that serve more than their config says (Nemotron 3.5 runs at 393K although its config says 256K). Only the default 32K context is now limited by the model's own maximum, so a small model isn't asked for more than it has.

### Added
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
