# Release notes

Newest first. Every commit adds its entry here.

## 0.1.2 (in progress)

### Added
- **CPU cores gauge** beside the ring gauges: one short bar per core, the same view as the CPU tile at the bottom (amber from 85%, red from 97%; hover a bar for its core and clock).
- **Network and Disk gauges** next to Temperature, GPU power and Load: network in and out, disk read and write, in MB/s. The rings use a square-root scale, so light traffic still shows.
- **Speed arrows** on the Stats block: Decode and Prefill show a green ▲ when the speed now is at least the average, an amber ▶ when a little slower, a red ▼ when well below it, and nothing while idle.
- **LiteLLM indicator** in the top bar next to Live: green when the gateway answers and accepts the key, red when it doesn't (hover for why). It opens Settings, Gateway.
- **Traffic stats** on the home page next to the gauges, in six compact figures: tokens in and out, requests, decode and prefill speed now and on average, and requests running (with those waiting), summed over the running models. Totals count from when each model started.
- **Extra LiteLLM settings editor** in Settings, Gateway: edit `litellm/extra.env` in the page (Save, or Save and apply, which restarts the gateway for a few seconds), see which lines are ignored and whether the saved file is live. No root editing needed.

### Changed
- The traffic figures sit at the right of the System card under a **Stats** heading in the same style as **System**.
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
