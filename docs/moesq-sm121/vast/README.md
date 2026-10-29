# MoESQ 35B (medium config) on a rented vast.ai box

Goal: produce a MoESQ checkpoint of Qwen3.6-35B-A3B (paired-4:8 sparse NVFP4, W4A4) for about $20, then benchmark it on the Spark.
Not yet useful to serve: the vLLM side still needs the 0.30 -> 0.31 rebase (see `../README.md`). Do the free speed gate first
(rebase + layer benchmark) if you want a go/no-go before spending.

Nothing in `run_on_box.sh` uses your vast.ai account key or touches billing. You do the account steps; the script does the work on the box.

## What is yours to do

1. **Regenerate your API key** (the one pasted into the chat is exposed): vast.ai console -> Account -> API Keys. On your Mac: `vastai set api-key <NEW_KEY>`.
2. **Add an SSH public key** to your vast.ai account (console -> Account -> SSH Keys). Use the Spark's key if you want the Spark to pull the result directly (`cat ~/.ssh/id_ed25519.pub` on the Spark).
3. **Pick an offer.** `python3 vast_search.py` prints live offers with an estimated total cost. Prices and ids change within hours.
   As of 2026-10-22 the best fits under $20 were: A100 SXM4 40 GB at about $0.79/h (est. $11.5), A100 SXM4 80 GB at about $1.04/h (est. $15),
   RTX PRO 6000 at about $1.4/h (est. $15.6). H100 SXM listed at $4.17+/h with these filters (est. $30): too much.
   Filters used: verified, on-demand, 1 GPU, reliability >= 0.98, >= 500 Mbps down, >= 200 GB disk, >= 64 GB RAM, CUDA >= 12.8. Prefer 30+ CPU cores.
4. **Rent it** in the console: the PyTorch template (any recent one), **disk 200 GB**, launch mode **SSH, direct** if offered, on-demand.
5. **Copy the script and run the phases** (`ssh -p PORT root@HOST` from the instance page; use `tmux` so a dropped connection does not kill the run):
   ```bash
   scp -P PORT run_on_box.sh root@HOST:/workspace/
   ssh -p PORT root@HOST
   cd /workspace && tmux new -s moesq
   bash run_on_box.sh setup                    # ~15-25 min, mostly the 72 GB download
   PRICE=0.79 bash run_on_box.sh probe         # prints projected hours and dollars for the full run
   PRICE=0.79 CAP=16 bash run_on_box.sh run    # only if the probe says it is within the cap
   ```
   If the probe projects more than the cap, **destroy the instance** (loss: under a dollar or two).
6. **Copy the result and destroy the instance.** The script prints the exact command at the end (rsync from the Spark, or scp to your Mac). Then `vastai destroy instance <ID>` or use the console. It keeps billing until you do.

## What the script does for you
- `setup`: picks cu128/cu130 torch wheels from the driver, builds a venv, installs MoESQ at the pinned commit (`584d78c`), flash-linear-attention, switches the Qwen3.5 wrapper from flash_attention_2 to sdpa, downloads the BF16 model and checks every shard against its header.
- `probe`: runs one layer of the real config, reports seconds per layer, projected hours and dollars at your PRICE, and the GPU memory peak.
- `run`: refuses to start if the probe projected more than CAP; every minute it prints spent and projected total, and **kills the run if the projection passes 115% of CAP** (and stops the instance if the platform's own container key is available).
  If the box dies mid-run, restart it and use `RESUME=1 PRICE=... bash run_on_box.sh run`.
- `pack`: assembles the checkpoint (`save_model.py`), writes `SHA256SUMS`.

## Budget (estimates, +-2x)
Setup and probe: about $1. Full run on an A100 SXM4 40 GB: about 10-16 h, $8-13 compute plus overhead. The cap is a guardrail, not a promise.

## Untested
The script was syntax-checked and its log parsing and config generation were tested locally against the real log format. It has **not run on a real box**: the first run is the test, which is why the probe comes first.
