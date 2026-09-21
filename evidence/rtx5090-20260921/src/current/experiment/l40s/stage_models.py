#!/usr/bin/env python3
"""Stage the SpeCo C5 fixture locally and push it to the GPU host in parallel.

The GPU host reaches Hugging Face through a relay at <0.5 MB/s; the local host reaches the
HF mirror at tens of MB/s and parallel scp streams reach ~12 MB/s.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

HF_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
HF_RESOLVE = "{base}/{repo}/resolve/{rev}/{path}"
HF_API = "{base}/api/models/{repo}/tree/{rev}?recursive=true"
REMOTE = "gj5090"


def token():
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        return open(path).read().strip()
    return os.environ.get("HF_TOKEN", "")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_files(repo, rev):
    headers = {"User-Agent": "curl/8"}
    if token():
        headers["Authorization"] = f"Bearer {token()}"
    for attempt in range(6):
        try:
            base = HF_ENDPOINTS[attempt % len(HF_ENDPOINTS)]
            url = HF_API.format(base=base, repo=repo, rev=rev)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                payload = json.load(response)
            return [
                {"path": e["path"],
                 "size": e.get("size") or (e.get("lfs") or {}).get("size", 0),
                 "sha256": (e.get("lfs") or {}).get("oid", "") if e.get("lfs") else ""}
                for e in payload if e.get("type") == "file" and e["path"] != ".gitattributes"
            ]
        except Exception as exc:  # noqa: BLE001 - retry loop
            print(f"[stage] listing retry {attempt} {repo}: {type(exc).__name__}", flush=True)
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"cannot list {repo}")


def download(url, target, size, want, attempts=400):
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    for _ in range(attempts):
        have = os.path.getsize(target) if os.path.exists(target) else 0
        if have == size:
            break
        if have > size:
            os.remove(target)
            have = 0
        headers = {"User-Agent": "curl/8"}
        if token():
            headers["Authorization"] = f"Bearer {token()}"
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response, \
                    open(target, "ab") as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except Exception:  # noqa: BLE001 - resume loop
            time.sleep(1)
            continue
    if os.path.getsize(target) != size:
        raise RuntimeError(f"size mismatch {target}")
    if want and sha256(target) != want:
        os.remove(target)
        raise RuntimeError(f"sha mismatch {target}")
    return target


def ssh(config, command, **kwargs):
    kwargs.setdefault("check", False)
    return subprocess.run(["ssh", "-F", config, REMOTE, command], **kwargs)


def push(spec, dest, remote_root, config, workers):
    name = spec["name"]
    remote_dir = f"{remote_root}/{name}"
    ssh(config, f"mkdir -p {remote_dir}", check=True)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        dirs = [pool.submit(ssh, config, f"mkdir -p {remote_dir}/{os.path.dirname(e['path'])}")
                for e in spec["files"] if os.path.dirname(e["path"])]
        for future in dirs:
            future.result()
        def push_one(entry):
            # Upload to a partial name and rename only after the byte count matches, so an
            # interrupted push never leaves a file that looks complete.
            remote_path = f"{remote_dir}/{entry['path']}"
            subprocess.run(["rsync", "-a", "--partial", "-e",
                            f"ssh -F {config} -o Compression=no",
                            os.path.join(dest, entry["path"]),
                            f"{REMOTE}:{remote_path}.partial"],
                           check=False)
            check = ssh(config, f"stat -c%s {remote_path}.partial 2>/dev/null || echo 0",
                        capture_output=True, text=True).stdout.strip()
            if check != str(entry["size"]):
                return entry["path"]
            ssh(config, f"mv {remote_path}.partial {remote_path}")
            return None

        futures = {pool.submit(push_one, e): e for e in spec["files"]}
        failed = [res for res in (f.result() for f in concurrent.futures.as_completed(futures)) if res]
    if failed:
        print(f"[stage] push failed for {len(failed)} files: {failed[:5]}", flush=True)
        return False
    print(f"[stage] {name}: pushed {sum(e['size'] for e in spec['files']) / 1e9:.2f} GB "
          f"in {time.time() - started:.0f}s", flush=True)
    verify = (
        "import hashlib,os,sys\n"
        f"root={remote_dir!r}\n"
        "files=" + json.dumps([[e["path"], e["size"], e.get("sha256", "")] for e in spec["files"]]) + "\n"
        "bad=[]\n"
        "for path,size,want in files:\n"
        "    p=os.path.join(root,path)\n"
        "    if not os.path.exists(p) or os.path.getsize(p)!=size:\n"
        "        bad.append((path,'size')); continue\n"
        "    if want:\n"
        "        h=hashlib.sha256()\n"
        "        with open(p,'rb') as fh:\n"
        "            for c in iter(lambda: fh.read(16*1024*1024), b''): h.update(c)\n"
        "        if h.hexdigest()!=want: bad.append((path,'sha'))\n"
        "print('verify bad:', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = ssh(config, f"~/0z5a/bin/python - <<'EOF'\n{verify}EOF", capture_output=True, text=True)
    print(f"[stage] {name}: {result.stdout.strip() or result.stderr.strip()}", flush=True)
    if result.returncode != 0:
        return False
    ssh(config, f"printf verified > {remote_dir}/.fetch-complete", check=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--remote-root", required=True)
    ap.add_argument("--ssh-config", default=os.path.expanduser("~/Documents/infra/.sshcm/config"))
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--push-workers", type=int, default=4)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    for spec in json.load(open(args.spec)):
        name, repo, rev = spec["name"], spec["repo"], spec.get("rev", "main")
        files = hf_files(repo, rev)
        drop = spec.get("drop", [])
        files = [f for f in files if f["size"] and not any(t in f["path"] for t in drop)]
        spec["files"] = files
        dest = os.path.join(args.stage, name)
        print(f"[stage] {name}: {len(files)} files {sum(f['size'] for f in files) / 1e9:.2f} GB", flush=True)
        if ssh(args.ssh_config, f"test -f {args.remote_root}/{name}/.fetch-complete").returncode == 0:
            print(f"[stage] {name}: remote already complete", flush=True)
            continue
        todo = [f for f in files
                if not (os.path.exists(os.path.join(dest, f["path"]))
                        and os.path.getsize(os.path.join(dest, f["path"])) == f["size"])]
        if todo:
            started = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
                futures = {
                    pool.submit(download, HF_RESOLVE.format(base=HF_ENDPOINTS[0], repo=repo, rev=rev,
                                                            path=f["path"]),
                                os.path.join(dest, f["path"]), f["size"], f.get("sha256", "")): f for f in todo
                }
                for future in concurrent.futures.as_completed(futures):
                    entry = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - abort this model
                        print(f"[stage] FAILED {entry['path']}: {type(exc).__name__} {exc}", flush=True)
                        return 1
                    done = sum(os.path.getsize(os.path.join(dest, f["path"]))
                               for f in files if os.path.exists(os.path.join(dest, f["path"])))
                    print(f"[stage] ok {entry['path']} ({done / 1e9:.2f}/{sum(f['size'] for f in files) / 1e9:.2f} GB, "
                          f"{done / max(time.time() - started, 1) / 1e6:.1f} MB/s)", flush=True)
        if not push(spec, dest, args.remote_root, args.ssh_config, args.push_workers):
            return 1
        if not args.keep:
            subprocess.run(["rm", "-rf", dest], check=False)


if __name__ == "__main__":
    sys.exit(main())
