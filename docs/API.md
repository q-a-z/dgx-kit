# DGX-kit HTTP API

Everything the dashboard does goes through this API, so you can script it. All paths start with `/api`, all bodies and answers are JSON (except logs, the live stream and errors noted below).

The dashboard also serves an interactive, always-current reference at `/docs` (Swagger) and `/openapi.json`, generated from the code.

## Conventions

**Base address:** `http://<host>:3000` (the dashboard port; see `DGXKIT_PORT`).

**Sign in.** When an admin password is set, every route except `POST /api/login` and `GET /api/me` needs the session cookie `dgxkit_session` (valid 7 days, HttpOnly). A call without it gets `401 {"detail": "sign in first"}`. Five wrong passwords from one address within five minutes give `429`.

```bash
curl -c jar -X POST localhost:3000/api/login -H 'Content-Type: application/json' -d '{"password":"…"}'
curl -b jar localhost:3000/api/models
```

**Errors** are `{"detail": …}` with a status code: `404` unknown name, `409` refused for a reason (`detail.problems` is a list of plain-language reasons), `422` invalid input, `403` read-only mode or wrong current password, `401` sign in first.

**Read-only mode** (`DGXKIT_READONLY=1`): everything that would change containers, images, downloads or models (starting, stopping, downloading, deleting, `/api/images*` except choosing a tag, `/api/gateway*`) answers `403`. Reading always works.

**Names.** A model name is its recipe name (letters, digits, `.`, `-`, `_`). Its container is `dgxkit-<name>`.

## Session

| Method and path | What it does |
|---|---|
| `GET /api/me` | Open to everyone. `{"auth_required", "signed_in", "readonly", "version"}` |
| `POST /api/login` | Body `{"password"}`. Sets the session cookie; `401` if wrong |
| `POST /api/logout` | Clears the cookie |
| `POST /api/password` | Body `{"old", "new"}` (`new` 8+ characters). Signs out every other session. With no password set, `old` may be empty and this sets the first one |

## Live data

| Method and path | What it does |
|---|---|
| `GET /api/snapshot` | The latest reading: GPU, CPU, memory, disk, and each running model's live numbers (`models`) |
| `GET /api/history` | The recent readings the charts draw, oldest first |
| `GET /api/stream` | Server-sent events: one `data: <snapshot JSON>` about every second, `: keep-alive` comments in between |
| `GET /api/layout`, `PUT /api/layout` | The dashboard's tile layout, shared by every browser. `PUT` body is the layout object |
| `GET /api/log` | What DGX-kit has done on this machine (the Activity tab): start, stop, create, password, … |

## Models

A model is a **recipe**: the engine, the weights, and its options. Fields (all optional except `name` and `repo`):

| Field | Meaning |
|---|---|
| `name`, `repo` | Its name here, and the Hugging Face repo |
| `engine` | `vllm` (default), `sglang` or `llamacpp` |
| `image` | The Docker image; empty uses the engine's default |
| `revision` | A repo revision |
| `quantization` | `nvfp4`, `fp8`, `gguf` or none (detected from the repo) |
| `gguf_file` | The file to load, for llama.cpp |
| `draft_repo`, `draft_method`, `num_speculative_tokens` | A speculative-decoding draft: `eagle3`, `dflash`, `mtp`, `draft_model` or `auto` |
| `speculative_extra` | More `--speculative-config` keys |
| `kv_cache_dtype`, `kv_cache_bytes` | KV cache type, and a fixed size (empty: sized for the context) |
| `fill_memory` | `true` takes every free byte for the KV cache and the longest context that fits (the old behaviour); default `false` sizes the KV cache for the context and `min_concurrency` only |
| `gpu_memory_utilization` | A fixed share of GPU memory (empty: worked out) |
| `max_context`, `min_context`, `min_concurrency` | Sizing targets. An empty `max_context` means 32768 (or the model's maximum if smaller) unless `fill_memory` is set |
| `extra_args` | Extra engine command-line arguments |
| `env` | Extra environment for the engine container |
| `docker` | Container settings: `mem_limit`, `shm_size`, `volumes` (`["src:dst[:mode]"]`) |
| `notes` | Your own notes |
| `path`, `draft_path` | Weights already on disk instead of a download |
| `publish` | List it on the LiteLLM gateway while it runs (default `true`) |

| Method and path | What it does |
|---|---|
| `GET /api/models` | Every model: the recipe fields, plus `options` (the vLLM options as one text block), `quick` (latest speed check), `container` (Docker state; `stopped_cleanly` true when you stopped it), `live` (its current numbers), `downloaded` |
| `POST /api/models` | Create from a recipe body. `201`; `409` if the name exists |
| `PUT /api/models/{name}` | Replace a recipe. The old one goes to the version history |
| `DELETE /api/models/{name}` | Forget the model. Its files on disk are never touched |
| `GET /api/models/{name}/versions` | Saved earlier versions of the recipe |
| `POST /api/models/{name}/restore/{version}` | Go back to one of them |
| `GET /api/models/{name}/plan` | The memory plan for the saved recipe: `context_tokens`, `kv_pool_tokens`, `kv_bytes`, `concurrency`, `fits`, `reason`, and `total_bytes` (weights, KV cache and headroom: what the model should take in all). A GGUF model is sized from its file's header |
| `POST /api/models/{name}/plan?own_bytes=N` | The same for unsaved edits in the body. `own_bytes` is memory the model holds now and would give back on restart |
| `POST /api/models/{name}/start` | Start it. Answers `{"port", "plan", "command"}`; `409` with `problems` when it can't start (not downloaded, GGUF file missing, not enough memory). When its engine image isn't on the box it answers `202 {"preparing": {image, state, progress, tail}}`: the image is pulled (or built, for a local build with no other option) and the model starts by itself afterwards; the model's row shows `preparing` meanwhile, and `stop` cancels. A local build nobody built is replaced by the engine's own image. The gateway is updated and a ten-second speed check runs once it answers |
| `POST /api/models/{name}/stop` | Stop it. `409` when requests are running or waiting, unless `?force=1` |
| `GET /api/models/{name}/logs?tail=200` | Container log tail (text) |
| `POST /api/models/{name}/publish` | Body `{"publish": true|false}`: list it on the gateway or not, without restarting it |
| `POST /api/models/{name}/download` | Download its weights from Hugging Face |
| `POST /api/recipes/inspect` | Body `{"repo", "draft_repo"?}`: read a repo's config and suggest a recipe |

### Templates

Ready-made sets of sizing settings (`compact`, `balanced`, `long-context`, `many-users`, `all-memory`, and your own).

| Method and path | What it does |
|---|---|
| `GET /api/templates` | All templates: `name`, `builtin`, `about`, `settings` |
| `POST /api/models/{name}/save-template` | Body `{"name", "about"}`: save this model's settings as a template. `201` |
| `POST /api/models/{name}/apply-template/{template}` | Apply it; the old settings stay in the version history |
| `DELETE /api/templates/{template}` | Delete your own template |

### Downloads

`{repo}` is a Hugging Face repo id such as `org/name` (slashes allowed).

| Method and path | What it does |
|---|---|
| `GET /api/downloads` | Downloads in progress, with progress |
| `POST /api/downloads/{repo}/pause`, `…/resume` | Pause or resume |
| `DELETE /api/downloads/{repo}` | Cancel |

### Benchmarks

| Method and path | What it does |
|---|---|
| `POST /api/models/{name}/bench` | Run `tools/bench.py` against a running model. Optional body chooses the tests (`decode`, `complex`, `hardcore`, `conc`, `prefill`, `stall`, `needle`, `tools`, `sanity`). `202` with the run |
| `GET /api/bench?model=name` | Runs, optionally for one model |
| `GET /api/bench/{run_id}` | One run, with results and output so far |
| `POST /api/bench/{run_id}/stop` | Stop a run |

## Model folders and import

| Method and path | What it does |
|---|---|
| `GET /api/library` | Models and drafts found in the model folders, set up or not |
| `GET /api/library/paths`, `PUT /api/library/paths` | The folders searched. Body `{"paths": [...]}` |
| `GET /api/library/explain?path=` | Why a folder is or isn't listed |
| `PUT /api/library/kind` | Body `{"path", "kind"}`: say whether a folder is a model or a draft (`null` for automatic) |
| `POST /api/library/setup` | Body `{"path", "draft_path"?, "name"?, "gguf_file"?}`: make a model from weights on disk. `201` |
| `GET /api/library/delete-preview?path=` | What deleting a folder from disk would remove |
| `POST /api/library/delete` | Body `{"path", "confirm"}`: **permanently** delete a model folder. `confirm` must be the typed confirmation text |
| `GET /api/import/llmctl?folder=` | What each llmctl `.conf` in a folder would become. Reads only |
| `POST /api/import/llmctl` | Body `{"files": [...]}`: save the chosen confs as models. Nothing starts |
| `GET /api/running` | Model servers on this machine that DGX-kit didn't start (view only) |
| `GET /api/running/{name}/launch` | How one of them was launched |
| `GET /api/running/{name}/logs?tail=200` | Its log tail |
| `POST /api/import/running/{name}` | Save it as a DGX-kit model from its launch command; the container keeps running |

## Gateway (LiteLLM)

| Method and path | What it does |
|---|---|
| `GET /api/gateway` | Status: `state`, `port`, `image`, `problem`, `db`, `key_ready`, `reachable`, `auth` (`rejected` when the key is wrong), `served` (model names), `url`, `extra_env` (names in effect), `extra_env_file` |
| `POST /api/gateway/sync` | Set up or repair the gateway: key, images, Postgres, LiteLLM. Safe to repeat. Recreates LiteLLM when its settings changed |
| `GET /api/gateway/extra` | The extra LiteLLM settings file: `file`, `text`, `applied` (names), `ignored` (`[{line, text, why}]`), `pending` (saved but not in the running gateway) |
| `PUT /api/gateway/extra` | Body `{"text", "apply"?}`: save the file; with `apply: true` also recreate the gateway now (a few seconds' restart; models keep running). `LITELLM_MASTER_KEY` and `DATABASE_URL` are ignored |
| `GET /api/settings/gateway` | The address and key settings: `url`, `url_default`, `key_set`, `key_hint` |
| `PUT /api/settings/gateway` | Body `{"url"?, "key"?, "clear_key"?}` |
| `GET /api/settings/gateway/key` | The key itself, for copying into a client |

Clients use the gateway directly at `http://<host>:4000/v1` with the key (OpenAI-compatible); it is a separate service from this API.

## Images

| Method and path | What it does |
|---|---|
| `GET /api/images` | Per engine: `image`, `default`, `ready`, `source`, and the local `builds` (`id`, `image`, `about`, `ready`, `patches`, `job`) |
| `PUT /api/images/{engine}` | Body `{"image": "tag" | null}`: point an engine at another tag; `null` returns to the pinned default. Applies at the next start |
| `GET /api/images/choices/{engine}` | Tags one model can be set to |
| `POST /api/images/{engine}/pull` | Pull the engine's image (builds it instead when it is a local build). `202` |
| `POST /api/images/pull` | Body `{"image": "registry/name:tag"}`: pull any tag, for example one picked in a model's settings. `202`; `422` for a local build tag, which has to be built |
| `GET /api/images/status?image=` | `{image, ready, build, job}`: whether the tag is on the box, the local build that makes it (if it is one), and the progress of a pull or build of it (`state`, `progress` 0..1, `tail`) |
| `POST /api/images/builds/{build}?make_default=false` | Build a local image (the GB10 vLLM images, llama.cpp). `202`. With `make_default=true` the engine switches to it |
| `POST /api/images/clean` | Remove engine images nothing uses |

`{engine}` is `vllm`, `sglang`, `llamacpp` or `litellm`.

## System

| Method and path | What it does |
|---|---|
| `GET /api/system` | `items` (model, vendor, BIOS, GPU VBIOS, kernel, driver, CUDA, Docker, DGX-kit, each with `group`, `label`, `value`), `firmware` (`checked`, `error`, `updates`, `devices` from fwupd with their `updates`), `changes` (what changed and when) and `since` |
| `GET /api/system/update` | `{current, latest, available, job}`: the running version; what GitHub has (`latest`: `version`, `sha`, `notes` of the newest release-notes section, `checked`, `repo`, or `error`; looked up at most once an hour); whether `latest` is newer; and the progress of an update (`job`: `state` running / restarting / failed, `step`, `tail`, `error`) |
| `POST /api/system/update/check` | Look at GitHub now |
| `POST /api/system/update` | Update from GitHub's `main`: download the source, build the image (the old one stays as `:previous`) and restart the dashboard. `202` with the state; `409` while one runs; `403` in read-only mode. Models and the gateway keep running. The page loses the connection for a moment at the end and comes back on the new version |
| `POST /api/system/firmware/check` | Ask fwupd for devices and updates now (a short-lived container with the system D-Bus). Reports only; installs nothing. `403` in read-only mode |

## Settings

| Method and path | What it does |
|---|---|
| `GET /api/settings/hf` | `{"set", "hint", "source"}`: is a Hugging Face token set, and where from |
| `PUT /api/settings/hf` | Body `{"token"?, "clear"?}`. A token is checked with Hugging Face before it is saved |
| `GET /api/settings/slo`, `PUT /api/settings/slo` | The latency targets the dashboard judges speed against: `ttft`, `itl`, `tpot`, `e2e` (seconds) |

## Example: start a model and wait for it

```bash
curl -b jar -X POST localhost:3000/api/models/nemotron-3.5/start
until curl -sb jar localhost:3000/api/models | python3 -c \
  'import json,sys; m=[x for x in json.load(sys.stdin) if x["name"]=="nemotron-3.5"][0]; sys.exit(0 if m["live"] else 1)'; do sleep 5; done
```
