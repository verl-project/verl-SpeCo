# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run Qwen3-8B DSpark train/save/cold-resume on two Ascend hosts.

Requires Paramiko on the controller, trusted SSH host keys, and a prepared
SpeCo / verl / CANN / Mooncake / vLLM environment on both remote hosts.
Use --help for paths and role assignments.
"""

import argparse
import json
import os
import secrets
import shlex
import sys
from contextlib import ExitStack
from getpass import getuser
from ipaddress import IPv4Address
from pathlib import Path
from uuid import uuid4

import paramiko


def connect(args, role):
    ip = str(getattr(args, role))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if args.known_hosts:
        client.load_host_keys(args.known_hosts)
    client.connect(
        ip,
        port=getattr(args, role + "_port"),
        username=args.user,
        password=os.getenv("SPECO_SSH_" + role.upper() + "_PASSWORD"),
        timeout=15,
    )
    client.get_transport().set_keepalive(20)
    return client


def execute(client, script, timeout=90):
    # Merge stderr before reading so a noisy command cannot fill its other pipe.
    with client.get_transport().open_session(timeout=10) as channel:
        channel.set_combine_stderr(True)
        channel.settimeout(timeout)
        channel.exec_command("bash -ec " + shlex.quote(script))
        channel.shutdown_write()
        with channel.makefile("rb") as stream:
            output = stream.read().decode(errors="replace").strip()
        if channel.recv_exit_status() != 0:
            raise RuntimeError(
                f"{client.get_transport().getpeername()[0]}: {output[-3000:]}"
            )
        return output


def start(client, directory, role):
    directory = shlex.quote(directory)
    command = (
        (
            "source ./job.env && exec python3 -m mooncake.http_metadata_server "
            '--host "$HOST_IP" --port 25501'
        )
        if role == "metadata"
        else f"exec bash {role}.sh"
    )
    execute(
        client,
        f"""
        cd {directory}
        nohup setsid env DSPARK_TWO_NODE_JOB={directory} bash -ec {shlex.quote(command)} > {role}.log 2>&1 </dev/null &
        echo $! > launcher-{role}.pid
    """,
    )


def wait_service(client, directory, role, probe, timeout):
    directory = shlex.quote(directory)
    execute(
        client,
        f"""
        cd {directory}
        deadline=$((SECONDS + {timeout}))
        until {probe}; do
            if ! kill -0 "$(cat launcher-{role}.pid)" 2>/dev/null || (( SECONDS >= deadline )); then
                tail -30 {role}.log; exit 1
            fi
            sleep 2
        done
    """,
        timeout=timeout + 15,
    )


def stop(client, directory):
    # Ray and vLLM children can create new process groups. Match the unique inherited
    # job tag on each process, including orphaned children, on this host only.
    execute(
        client,
        f"""python3 - {shlex.quote(directory)} <<'PY'
import errno, os, signal, sys, time
from pathlib import Path
tag = b'DSPARK_TWO_NODE_JOB=' + os.fsencode(sys.argv[1])
def belongs_to_job(pid):
    try:
        return tag in Path('/proc', str(pid), 'environ').read_bytes().split(b'\\0')
    except (ProcessLookupError, FileNotFoundError, PermissionError):
        return False
def running():
    for path in Path('/proc').glob('[0-9]*/environ'):
        pid = int(path.parent.name)
        if belongs_to_job(pid):
            yield pid
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in running():
        pidfd = None
        try:
            if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                try:
                    pidfd = os.pidfd_open(pid)
                except OSError as exc:
                    if exc.errno not in (errno.ENOSYS, errno.EINVAL):
                        raise
            if belongs_to_job(pid):
                if pidfd is None:
                    os.kill(pid, sig)
                else:
                    signal.pidfd_send_signal(pidfd, sig)
        except ProcessLookupError:
            pass
        finally:
            if pidfd is not None:
                os.close(pidfd)
    for _ in range(25):
        if not any(running()):
            sys.exit(0)
        time.sleep(0.2)
raise SystemExit('Some tagged processes are still running')
PY""",
    )


def stop_phase(consumer, consumer_dir, producer, producer_dir):
    # A lost reader can still access registered source memory. Never release the
    # source unless every task-owned Consumer process is confirmed stopped.
    try:
        stop(consumer, consumer_dir)
    except Exception as error:
        raise RuntimeError(
            f"Consumer shutdown was not confirmed; preserving Producer resources at {producer_dir}. "
            "Fence all readers before stopping the Producer."
        ) from error
    stop(producer, producer_dir)


def setup_script(args, role):
    path = getattr(args, role + "_env")
    return "source " + shlex.quote(path) + "\n" if path else ""


def check(client, args, role):
    device = getattr(args, role + "_npu")
    execute(
        client,
        setup_script(args, role)
        + f"""
        test -d {shlex.quote(args.root)}
        test -r {shlex.quote(args.repo + "/examples/run_qwen3-8b_drafter_dspark_separate_training.sh")}
        test -r {shlex.quote(args.model + "/config.json")} && test -r {shlex.quote(args.data)}
        mapping=$(npu-smi info -m)
        if ! chip=$(awk -v device={device} '
            $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $3 == device {{
                card = $1; chip_id = $2; matches++
            }}
            END {{ if (matches == 1) print card, chip_id; else exit 1 }}
        ' <<< "$mapping"); then
            echo "Cannot map logical NPU {device} to a unique chip"; exit 1
        fi
        read -r card chip_id <<< "$chip"
        if ! usage=$(npu-smi info -t proc-mem -i "$card" -c "$chip_id") ||
            ! grep -Eq '^[[:space:]]*No process in device[.][[:space:]]*$' <<< "$usage"; then
            echo "Logical NPU {device} is busy or its process status is unavailable"; exit 1
        fi
        if ! health=$(npu-smi info -t health -i "$card" -c "$chip_id") ||
            ! grep -Eq 'Health Status[[:space:]]*:[[:space:]]*OK[[:space:]]*$' <<< "$health"; then
            echo "Logical NPU {device} is unhealthy or its health status is unavailable"; exit 1
        fi
        ports=$(ss -H -ltn | awk '{{print $4}}')
        if grep -Eq ':(2549[0-7]|2550[1-4]|255[23][0-9]|2554[0-5])$' <<< "$ports"; then
            echo "A service port is in use"; exit 1
        fi
    """,
    )


def prepare(client, args, role, name, token, generation):
    directory = f"{args.root}/{name}-{role}"
    execute(client, f"umask 077; mkdir {shlex.quote(directory)}")
    source_slots, reader_bytes = 4, 512 << 20
    variables = {
        "ROLE": role,
        "RUN_DIR": directory,
        "REPO": args.repo,
        "MODEL": args.model,
        "HOST_IP": str(getattr(args, role)),
        "PRODUCER_IP": str(args.producer),
        "CONSUMER_NPU": str(args.consumer_npu),
        "PRODUCER_NPU": str(args.producer_npu),
        "SPECO_EXTERNAL_RAY_ADDRESS": f"{args.consumer}:25490",
        "SPECO_VLLM_ENDPOINTS": json.dumps([f"http://{args.producer}:25504/v1"]),
        "SPECO_VLLM_HIDDEN_STATES_STORE": "native",
        "SPECO_NATIVE_RUN_ID": name,
        "SPECO_NATIVE_DIRECTORY": f"native_{name}",
        "SPECO_NATIVE_GENERATION": str(generation),
        "SPECO_NATIVE_RAY_ADDRESS": f"{args.consumer}:25490",
        "SPECO_NATIVE_RAY_NAMESPACE": "transfer_queue",
        "SPECO_NATIVE_SOURCE_ENDPOINT": f"{args.producer}:25502",
        "SPECO_NATIVE_READER_ENDPOINTS": json.dumps([f"{args.consumer}:25503"]),
        "SPECO_NATIVE_METADATA_SERVER": f"http://{args.consumer}:25501/metadata",
        "SPECO_NATIVE_SOURCE_MAX_SLOTS": str(source_slots),
        # 8192 tokens x 4096 hidden width x 6 BF16 layers need 384 MiB,
        # before token IDs, descriptors and registration alignment.
        "SPECO_NATIVE_SOURCE_MAX_BYTES": str(source_slots * reader_bytes),
        "SPECO_NATIVE_READER_MAX_BYTES": str(reader_bytes),
        "SPECO_NATIVE_REUSE_BUFFER_BYTES": str(448 << 20),
        "SPECO_NATIVE_CHUNK_BYTES": str(128 << 20),
        "SPECO_NATIVE_MAX_RECORDS": "128",
        "MC_LEGACY_RPC_PORT_BINDING": "1",
        "HCCL_IF_IP": str(getattr(args, role)),
        "DRAFT_CKPTS_DIR": f"{args.root}/{name}-consumer/checkpoints",
        "MC_METADATA_HTTP_TOKEN": token,
    }
    env = setup_script(args, role) + "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in variables.items()
    )
    env += 'export PYTHONPATH="$REPO:${PYTHONPATH:-}"\n'
    env += f'export NO_PROXY="127.0.0.1,localhost,{args.consumer},{args.producer},${{NO_PROXY:-}}"\nexport no_proxy="$NO_PROXY"\n'
    helpers = Path(__file__).resolve().parents[2] / "tools" / "dspark_two_node"
    with client.open_sftp() as sftp:
        for filename in ("train.sh", "ray.sh", "vllm.sh", "job.env"):
            contents = (
                env
                if filename == "job.env"
                else (helpers / filename).read_text(encoding="utf-8")
            )
            with sftp.open(f"{directory}/{filename}", "wb") as dest:
                dest.write(contents.encode())
    execute(client, f"chmod 600 -- {shlex.quote(directory + '/job.env')}")
    return directory


def phase(args, hosts, name, checkpoint, resume):
    token = secrets.token_urlsafe(32)
    consumer, producer = hosts
    c, p = [
        prepare(host, args, role, name, token, int(resume))
        for host, role in zip(hosts, ("consumer", "producer"), strict=True)
    ]
    action = "resume" if resume else "train"
    print(f"{action}: {args.producer}:{p}/{action}.log", flush=True)
    with ExitStack() as cleanup:
        cleanup.callback(stop_phase, consumer, c, producer, p)
        start(consumer, c, "ray")
        start(consumer, c, "metadata")
        start(producer, p, "vllm")
        wait_service(consumer, c, "ray", "ss -H -ltn 'sport = :25490' | grep -q .", 90)
        start(producer, p, "ray")
        wait_service(
            consumer, c, "metadata", "ss -H -ltn 'sport = :25501' | grep -q .", 90
        )
        wait_service(
            producer,
            p,
            "vllm",
            f"curl --noproxy '*' -fsSo /dev/null --max-time 3 http://{args.producer}:25504/health 2>/dev/null",
            300,
        )
        execute(
            producer,
            f"""cd {shlex.quote(p)}; source ./job.env; python3 - <<'PY'
import os, ray, time
ray.init(address=os.environ["SPECO_EXTERNAL_RAY_ADDRESS"], logging_level="ERROR")
for _ in range(60):
    nodes = {{n["NodeManagerAddress"]: n["Resources"].get("NPU", 0) for n in ray.nodes() if n["Alive"]}}
    if nodes == {{"{args.consumer}": 1, "{args.producer}": 0}}:
        break
    time.sleep(1)
else:
    raise SystemExit(f"Unexpected Ray node resources: {{nodes}}")
ray.shutdown()
PY""",
        )
        log = f"{action}.log"
        execute(
            producer,
            f"""cd {shlex.quote(p)}
            if ! timeout 30m env DSPARK_TWO_NODE_JOB={shlex.quote(p)} bash train.sh {action} {shlex.quote(args.data)} {shlex.quote(checkpoint)} >{log} 2>&1; then
                tail -30 {log}; exit 1
            fi
            grep -Eq 'optimizer_step_total={50 if resume else 25} ' {log}
            grep -q 'pipeline completed consumer_steps=25 tail_dropped=0' {log}
        """,
            timeout=1830,
        )
        logs = '"$(cat ray.tmp)"/session_latest/logs/worker*.err'
        for host, path, expect, exclude in (
            (consumer, c, "StandaloneDrafterWorker", "StandaloneProducerActor"),
            (producer, p, "StandaloneProducerActor", "StandaloneDrafterWorker"),
        ):
            execute(
                host,
                f"""cd {shlex.quote(path)}
                grep -q '^:actor_name:{expect}' {logs} && ! grep -q '^:actor_name:{exclude}' {logs}""",
            )
        if resume:
            execute(
                consumer,
                f"""cd {shlex.quote(c)}; grep -Eq 'optimizer_steps_total=25 .*optimizer_state=restored' {logs}""",
            )
            execute(
                producer, f"cd {shlex.quote(p)}; grep -q 'consumed=50 step=25' {log}"
            )
        else:
            execute(
                producer,
                f"test -s {shlex.quote(checkpoint + '/optimizer/.metadata')} && test -s {shlex.quote(checkpoint + '/model.safetensors')}",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("producer", "consumer"):
        parser.add_argument(
            f"--{role}", type=IPv4Address, required=True, help="SSH and Ray node IP"
        )
        parser.add_argument(f"--{role}-port", type=int, default=22, help="SSH port")
        parser.add_argument(
            f"--{role}-npu", type=int, default=0, help="local Ascend device index"
        )
        parser.add_argument(
            f"--{role}-env",
            help="optional host-local shell file for CANN and Python environment",
        )
    parser.add_argument(
        "--user", help="SSH user on both hosts (defaults to the local user)"
    )
    parser.add_argument(
        "--known-hosts", help="additional trusted OpenSSH known_hosts file"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="same absolute path to this checkout on both hosts",
    )
    parser.add_argument(
        "--model", required=True, help="Qwen3-8B model path on both hosts"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="same input path on both hosts, with at least 100 usable texts",
    )
    parser.add_argument(
        "--root", required=True, help="absolute writable job root on both hosts"
    )
    parser.add_argument(
        "--consumer-view",
        help="path on both hosts exposing the Consumer job root (default: --root)",
    )
    parser.add_argument("--check", action="store_true", help="preflight only")
    args = parser.parse_args()
    if args.user is None:
        try:
            args.user = getuser()
        except (OSError, ImportError):
            parser.error("--user is required when the local username is unavailable")
    if args.consumer == args.producer:
        parser.error("Producer and Consumer must use distinct hosts")
    if not all(
        p.startswith("/")
        for p in (
            args.repo,
            args.model,
            args.data,
            args.root,
            args.consumer_view or args.root,
        )
    ):
        parser.error("remote paths must be absolute Linux paths")
    with ExitStack() as stack:
        hosts = [
            stack.enter_context(connect(args, role))
            for role in ("consumer", "producer")
        ]
        with hosts[0].get_transport().open_session() as lock:
            lock.exec_command(
                f"flock -n {shlex.quote(args.root + '/.dspark-two-node.lock')} sh -c 'echo locked; cat >/dev/null'"
            )
            if lock.makefile("rb").readline().strip() != b"locked":
                raise RuntimeError("Could not acquire the Consumer job lock")
            for host, role in zip(hosts, ("consumer", "producer"), strict=True):
                check(host, args, role)
            print("Both hosts ready", flush=True)
            if args.check:
                return
            name = f"dspark-{uuid4().hex[:12]}"
            checkpoint = f"{args.consumer_view or args.root}/{name}t-consumer/checkpoints/draft_step_25"
            phase(args, hosts, name + "t", checkpoint, False)
            phase(args, hosts, name + "r", checkpoint, True)
            print(f"50 optimizer steps completed. Checkpoint: {checkpoint}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, paramiko.SSHException) as error:
        sys.exit(str(error))
