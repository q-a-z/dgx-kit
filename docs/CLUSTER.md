# Running one model on two Sparks

DGX-kit can serve a model across two (or more) Sparks with vLLM's multi-node tensor parallelism. The head (rank 0) runs on
the Spark that runs DGX-kit and serves the API, so the LiteLLM gateway publishes it like any other model. Each other Spark
runs a headless worker container that DGX-kit starts over SSH.

**DGX-kit builds the commands and checks the requirements below. Setting them up is yours:** the fiber link, the SSH key,
and having the same image and weights on every node. Checkpoint reads have been verified on a two-Spark NFS setup.

## Requirements

1. **A fiber (or DAC) link between the two Sparks**, the same physical port on both, with an address on each side (below).
2. **SSH key login from the DGX-kit container to each worker**, with the worker's host key already in `known_hosts`.
   DGX-kit uses the keys and known-host entries under `DGXKIT_HOME/.ssh`, visible through the service's read-only
   home mount. The worker's SSH user must be allowed to use Docker.
3. **The same vLLM image on every node** (`docker image inspect` finds it on each).
4. **The same model folder at the same path on every node** (DGX-kit does not copy weights).
5. **An image built with this version**: the Dockerfile now installs `openssh-client` and `iproute2`, so rebuild and restart dgx-kit.

## Fiber link setup

Give each Spark an address on the port you cabled. This install uses `enp1s0f1np1`, `192.168.100.10` on the head and
`192.168.100.11` on the worker; the other port, `enp1s0f0np0`, is unused. Netplan, on each node with its own address:

    network:
      version: 2
      ethernets:
        enp1s0f1np1:
          addresses: [192.168.100.10/24]   # .11 on the worker
          mtu: 9000

Then `sudo netplan apply`, and check on both nodes:

    ip -br addr show enp1s0f1np1                          # UP, with the address
    cat /sys/class/net/enp1s0f1np1/speed                  # 200000 for a 200G link
    ping -M do -s 8972 192.168.100.11                     # jumbo frames pass (from the head)
    rdma link                                             # which RoCE device sits on the port, e.g. rocep1s0f1: ACTIVE

NCCL needs the RoCE device name (`hca`) and the GID index of the RoCE v2 IPv4 address:

    for i in 0 1 2 3; do echo $i $(cat /sys/class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/$i) $(cat /sys/class/infiniband/rocep1s0f1/ports/1/gids/$i); done

Pick the index whose type is `RoCE v2` and whose address is the IPv4-mapped one (3 here).

## The recipe

Add a `cluster` block to the model (there is no dashboard page for it yet: `PUT /api/models/<name>` with a `cluster`
field, or edit the model's YAML in the state folder):

    cluster:
      head_ip: 192.168.100.10
      interface: enp1s0f1np1
      hca: rocep1s0f1
      gid_index: 3
      master_port: 29501          # optional
      workers:
      - host: q@192.168.100.11    # the SSH target
        ip: 192.168.100.11        # its address on the fiber link

The engine must be vLLM. DGX-kit adds `--tensor-parallel-size`, `--nnodes`, `--node-rank` and `--master-addr` to the head,
starts the workers with `--headless` and their own rank, sets `VLLM_HOST_IP` and the NCCL variables per node, and gives
every container `/dev/infiniband`. The sizing plan divides the weights by the number of nodes.

## Checking

    curl -b "dgxkit_session=..." http://<spark>:3000/api/models/<name>/cluster

The JSON response lists each check and its result. Before a vLLM start, DGX-kit checks available memory on each worker
and opens checkpoint and draft files from a disposable container using the selected image and launch user.
The probe uses read-only model mounts and no GPU. A failed check blocks the launch. MTU, link-speed, and RDMA
warnings can still allow a start, and NCCL may fall back to TCP.

Stop cancels a launch that's still in preflight. If a rank cannot be stopped, the dashboard shows **Stop incomplete**
with the affected host and lets you retry. The Logs tab includes worker log tails so a startup error on a worker
can be traced from the head.

## Limits

- The check cannot prove NCCL works; the first start does. Read the head's log if it hangs at "init".
- DGX-kit does not poll the workers while the model runs. If a worker dies, the head normally fails with it; if it hangs, stop the model.
- The head must be the Spark that runs DGX-kit.
