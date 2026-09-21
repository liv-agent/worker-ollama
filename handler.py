import hashlib
import json
import os
import re
import subprocess
import time
import uuid

import requests
import runpod

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
OLLAMA_MODELS_DIR = os.environ.get("OLLAMA_MODELS", "/root/.ollama/models")

# Hugging Face inputs. These are NOT the same thing as OLLAMA_MODEL: HF_MODEL is a
# Hugging Face repo id, OLLAMA_MODEL is an Ollama model reference. HF_MODEL wins.
HF_MODEL_RAW = os.environ.get("HF_MODEL", "").strip()
HF_QUANTIZATION_RAW = os.environ.get("HF_QUANTIZATION", "").strip()
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "").strip()
# Where Runpod's model store mounts its prefilled Hugging Face cache. There is no
# env var injected by the platform to discover this, so it is a documented path
# with an override in case it moves.
RUNPOD_MODEL_CACHE_DIR = os.environ.get(
    "RUNPOD_MODEL_CACHE_DIR", "/runpod-volume/huggingface-cache/hub"
)
OLLAMA_TEMPLATE = os.environ.get("OLLAMA_TEMPLATE", "")
# Used only when neither HF_MODEL nor OLLAMA_MODEL is set, so the worker still
# serves something out of the box rather than erroring on the first request.
FALLBACK_MODEL = "llama3.2:3b"

# Quantization tokens as they appear in GGUF filenames. The boundary sets matter:
# quant tokens contain underscores, so splitting on "_" would be wrong, and a
# trailing "_" must not match (otherwise "Q4" would select "Q4_K_M").
QUANT_RE = re.compile(
    r"(?:^|[-_./])(I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32|MXFP4)(?=[-./]|$)", re.I
)
# GGUFs that live in a model repo but carry no language-model weights. A
# multimodal projector is the classic case: it ships alongside the model and is
# usually the *smallest* file in the repo, so a naive "smallest wins" default
# picks it and every request then fails with an immediate 400.
NON_MODEL_GGUF_RE = re.compile(r"(?:^|[-_.])(mmproj|mm-proj|projector)", re.I)
# Ollama's own default when a Hugging Face reference carries no tag, and the
# quantization model cards assume. Preferred over "smallest" because large repos
# start at 1-bit quants: unsloth/Qwen3-8B-GGUF's smallest is UD-IQ1_S (2.3 GB)
# against Q4_K_M's 5.0 GB, and IQ1 output quality is not usable for most work.
PREFERRED_QUANT = "Q4_K_M"
# Mirrors Ollama's own splitGGUFNameRe. Matching a different pattern than the
# server does is how you end up with a silently broken multi-layer manifest.
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)

session = requests.Session()

# Accepted shapes for the repo id, longest prefix first so 'https://hf.co/' is
# not partially matched by 'hf.co/'.
HF_REPO_PREFIXES = (
    "https://huggingface.co/",
    "http://huggingface.co/",
    "https://hf.co/",
    "http://hf.co/",
    "huggingface.co/",
    "hf.co/",
)


def parse_hf_model(value):
    """Split an HF_MODEL value into (repo_id, quantization).

    The Hub's Hugging Face picker stores this as 'hf.co/<org>/<repo>', users
    paste bare repo ids and browser URLs, and Ollama-style references carry the
    quantization as a ':tag'. Normalise all of them to a bare 'org/repo', since
    that is what both the model-store folder name and the HF API expect.
    """
    value = (value or "").strip()
    lowered = value.lower()
    for prefix in HF_REPO_PREFIXES:
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            break

    # A Hugging Face repo id can't contain ':', so a colon is always a tag.
    repo, _, tag = value.partition(":")

    # Keep only '<org>/<repo>', dropping web-UI suffixes like '/tree/main'.
    parts = [part for part in repo.split("/") if part][:2]
    repo = "/".join(parts)

    tag = tag.strip().strip("/")
    if tag.lower() == "latest":
        tag = ""
    return repo, tag


# Derived here rather than at the top so an 'hf.co/org/repo:Q4_K_M' style value
# works as well as a bare repo id. An explicit HF_QUANTIZATION still wins.
HF_MODEL, _HF_MODEL_TAG = parse_hf_model(HF_MODEL_RAW)
HF_QUANTIZATION = HF_QUANTIZATION_RAW or _HF_MODEL_TAG


def free_bytes(path):
    """Free space on the filesystem holding `path`, walking up to the nearest
    existing ancestor so it works before the directory is created."""
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    stat = os.statvfs(path or "/")
    return stat.f_bavail * stat.f_frsize


def require_free_space(path, needed, what):
    """Fail before doing 20 GiB of I/O that can only end in ENOSPC.

    Ollama re-writes a GGUF when registering it, so a plain import needs about
    twice the model size on the filesystem holding the blob store, and three
    times it when the file has to be uploaded instead of hard-linked.
    """
    available = free_bytes(path)
    if available >= needed:
        return
    raise ValueError(
        f"Not enough disk space for {what}: need {_human_size(needed)} free on "
        f"{path}, but only {_human_size(available)} is available. Increase the "
        f"endpoint's container disk, or attach a network volume so models are "
        f"stored there instead. Registering a GGUF needs roughly 3x the model "
        f"size at peak because Ollama re-writes the file."
    )


def is_out_of_space(message):
    return any(
        marker in (message or "").lower()
        for marker in ("no space left", "enospc", "disk full", "out of space")
    )


def gpu_total_bytes():
    """Total VRAM across visible GPUs, or 0 when it can't be determined.

    Ollama exposes no API for this, so shell out to nvidia-smi and treat any
    failure as "unknown" rather than as "no GPU".
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return 0
        return sum(int(line) * (1 << 20) for line in out.stdout.split() if line.strip().isdigit())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def vram_warning(model_bytes, vram_bytes, overhead=1.15):
    """Message to log when the weights won't comfortably fit in VRAM, else None.

    Deliberately advisory: Ollama offloads whatever doesn't fit to CPU, so the
    model still answers, just slowly. Failing hard here would break setups that
    work.
    """
    if not model_bytes or not vram_bytes:
        return None
    needed = model_bytes * overhead
    if needed <= vram_bytes:
        return None
    return (
        f"WARN: this model needs about {_human_size(needed)} of VRAM "
        f"({_human_size(model_bytes)} of weights plus ~{int((overhead - 1) * 100)}% for "
        f"KV cache and activations) but the worker has {_human_size(vram_bytes)}. "
        f"Ollama will offload the remainder to CPU, which is much slower. Pick a GPU "
        f"with more VRAM, a smaller quantization via HF_QUANTIZATION, or a lower "
        f"OLLAMA_CONTEXT_LENGTH."
    )


def ollama_error(response):
    """Ollama's reason lives in the response body; the status line says nothing.

    Returns a string combining status and body, or "" when the response is fine.
    """
    if response.ok:
        return ""
    detail = response.text.strip()
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            detail = str(payload["error"])
    except ValueError:
        pass
    return f"HTTP {response.status_code} from {response.request.path_url}: {detail[:600]}"


_described = set()


def describe_model(model):
    """Log what Ollama thinks this model can do, once per process per model.

    The first thing worth knowing when a request is rejected before the model is
    even loaded. Memoised because it sits on the hot path: without the guard it
    costs an /api/show round trip and a log line on every single request.
    """
    if model in _described:
        return
    try:
        response = session.post(f"{OLLAMA_BASE_URL}/api/show", json={"model": model}, timeout=60)
        if not response.ok:
            print(f"WARN: /api/show failed for '{model}': {ollama_error(response)}", flush=True)
            return
        info = response.json()
        template = (info.get("template") or "").strip()
        details = info.get("details") or {}
        print(
            f"Model '{model}': capabilities={info.get('capabilities')} "
            f"family={details.get('family')} params={details.get('parameter_size')} "
            f"quant={details.get('quantization_level')} template={'yes' if template else 'NO'}",
            flush=True,
        )
        if not template:
            print(
                f"WARN: '{model}' has no chat template embedded in the GGUF. Chat "
                f"responses may be malformed — set OLLAMA_TEMPLATE, or pass 'template' "
                f"in the request input.",
                flush=True,
            )

        # Size comes from /api/tags, which reports it per model; /api/show does not.
        size = 0
        tags = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
        if tags.ok:
            for entry in tags.json().get("models", []):
                if entry.get("name") in (model, f"{model}:latest"):
                    size = entry.get("size") or 0
        warning = vram_warning(size, gpu_total_bytes())
        if warning:
            print(warning, flush=True)

        _described.add(model)
    except (requests.RequestException, ValueError) as err:
        print(f"WARN: could not describe '{model}': {err}", flush=True)


def get_local_models():
    response = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def normalize_model_name(model):
    """Ollama only knows the 'hf.co' registry host.

    'huggingface.co/...' parses as a different host, so its manifest would be
    written somewhere Ollama never looks it up.
    """
    model = (model or "").strip()
    if model.startswith("huggingface.co/"):
        return "hf.co/" + model[len("huggingface.co/") :]
    return model


def _sanitize(part):
    """Ollama name parts allow [A-Za-z0-9_.-], but lookups fold case while the
    on-disk manifest path preserves it, so lowercase is the only safe form."""
    part = re.sub(r"[^a-z0-9._-]+", "-", part.lower()).strip("-._")
    return part or "model"


def derive_model_name(repo_id, quantization, model_file):
    """The Ollama name a Hugging Face repo is registered under.

    Pure function of the env inputs on purpose: start.sh prepares the model in a
    short-lived subprocess and then execs the handler as a separate process, so
    both must derive the same name without sharing state. In particular the tag
    comes from the *inputs*, never from whichever file selection picked, because
    selection needs to list the repo.
    """
    org, _, name = repo_id.strip("/").partition("/")
    base = _sanitize(f"{org}-{name}" if name else org)
    if len(base) > 55:
        digest = hashlib.sha256(repo_id.encode()).hexdigest()[:6]
        base = base[:48].rstrip("-._") + "-" + digest
    if model_file:
        tag = _sanitize(re.sub(r"\.gguf$", "", os.path.basename(model_file), flags=re.I))
    elif quantization:
        tag = _sanitize(quantization)
    else:
        tag = "latest"
    return f"hf/{base}:{tag[:60].rstrip('-._') or 'latest'}"


def resolve_default_model():
    """The single place HF_MODEL's precedence over OLLAMA_MODEL is decided.

    Precedence: HF_MODEL, then OLLAMA_MODEL, then FALLBACK_MODEL.
    """
    if HF_MODEL:
        return derive_model_name(HF_MODEL, HF_QUANTIZATION, HF_MODEL_FILE)
    return normalize_model_name(DEFAULT_MODEL) or FALLBACK_MODEL


def _hf_cache_roots():
    """Runpod's prefilled model store first.

    That mount may be read-only, so the read path is deliberately separate from
    the directory fallback downloads are written to.
    """
    roots = []
    for root in (
        RUNPOD_MODEL_CACHE_DIR,
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
    ):
        if root and root not in roots and os.path.isdir(root):
            roots.append(root)
    return roots


def _resolve_repo_folder(root, folder):
    """Match the cache folder case-insensitively.

    Hugging Face repo ids are case-sensitive but resolve case-insensitively via a
    307, so 'org/repo-gguf' downloads fine while Runpod's model store prefills
    under the canonical 'org/Repo-GGUF'. An exact-match-only lookup silently
    misses the prefilled copy and re-downloads the whole model.
    """
    if os.path.isdir(os.path.join(root, folder)):
        return folder
    wanted = folder.lower()
    try:
        for entry in sorted(os.listdir(root)):
            if entry.lower() == wanted and os.path.isdir(os.path.join(root, entry)):
                print(
                    f"[ModelStore] Matched '{folder}' to '{entry}' (case differs — set "
                    f"HF_MODEL to the repo's exact casing to avoid this lookup)",
                    flush=True,
                )
                return entry
    except OSError:
        pass
    return None


def find_cached_snapshot(repo_id):
    """Locate a Hugging Face hub snapshot dir, as Runpod's model store lays it out."""
    folder = "models--" + repo_id.strip("/").replace("/", "--")
    for root in _hf_cache_roots():
        resolved = _resolve_repo_folder(root, folder)
        if resolved is None:
            continue
        snapshots = os.path.join(root, resolved, "snapshots")
        if not os.path.isdir(snapshots):
            continue
        ref = os.path.join(root, resolved, "refs", "main")
        if os.path.isfile(ref):
            with open(ref) as f:
                candidate = os.path.join(snapshots, f.read().strip())
            if os.path.isdir(candidate):
                return candidate
        versions = sorted(
            d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))
        )
        if versions:
            return os.path.join(snapshots, versions[0])
    return None


def list_snapshot_files(snapshot_dir):
    """Repo-relative paths of files that are really there.

    Snapshot entries are symlinks into the cache's blobs/ dir, and a partially
    filled cache leaves dangling ones behind.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(snapshot_dir, followlinks=True):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if os.path.isfile(os.path.realpath(full)):
                found.append(os.path.relpath(full, snapshot_dir))
    return sorted(found)


def _human_size(num_bytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.0f} {unit}" if unit in ("B", "KiB") else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def _stem(path):
    return re.sub(r"\.gguf$", "", path, flags=re.I)


def _quant_pattern(quantization):
    """Match a quantization token on -, _, . or / boundaries.

    The trailing set deliberately excludes "_" so "Q4" does not match "Q4_K_M".
    """
    return re.compile(rf"(?:^|[-_./]){re.escape(quantization.strip())}(?=[-./]|$)", re.I)


def _shard_key(path):
    match = SHARD_RE.match(os.path.basename(path))
    if not match:
        return None
    return (os.path.dirname(path), match.group(1), match.group(3))


def available_quants(gguf_files):
    """Quantizations present in a repo, for error messages that tell users what to pick."""
    return sorted(
        {m.group(1).upper() for f in gguf_files for m in QUANT_RE.finditer(_stem(f))}
    )


def group_ggufs(gguf_files):
    """Collapse split-GGUF shards into one entry each, preserving listing order.

    Grouping is deliberately lenient here — an incomplete shard set for one
    quantization must not break selecting a different one.
    """
    groups = []
    index_of = {}
    for path in gguf_files:
        key = _shard_key(path)
        if key is None:
            groups.append([path])
            continue
        if key not in index_of:
            index_of[key] = len(groups)
            groups.append([])
        groups[index_of[key]].append(path)
    return groups


def validate_group(group, repo_id):
    """Order and completeness-check a selected group.

    Ollama needs every shard of a split GGUF, keyed by the exact filename
    llama.cpp wrote; handing it only the first shard is an error.
    """
    if _shard_key(group[0]) is None:
        return group
    total = int(SHARD_RE.match(os.path.basename(group[0])).group(3))
    found = {}
    for path in group:
        index = int(SHARD_RE.match(os.path.basename(path)).group(2))
        if index in found:
            raise ValueError(
                f"'{repo_id}' has duplicate shard {index} for "
                f"'{os.path.basename(group[0])}'."
            )
        found[index] = path
    missing = [i for i in range(1, total + 1) if i not in found]
    if missing:
        raise ValueError(
            f"Split GGUF '{os.path.basename(group[0])}' in '{repo_id}' is incomplete — "
            f"missing shard(s) {missing} of {total}. Ollama needs every shard."
        )
    return [found[i] for i in range(1, total + 1)]


def select_gguf(files, quantization, model_file, repo_id, sizes=None):
    """Repo-relative path(s) of the GGUF to load: one file, or every shard of a split GGUF.

    `sizes` maps repo-relative paths to byte sizes and is what makes the
    no-quantization default possible: without an explicit HF_QUANTIZATION the
    smallest variant wins.
    """
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        raise ValueError(
            f"Hugging Face repo '{repo_id}' contains no .gguf files, so Ollama cannot "
            f"run it. HF_MODEL must point at a GGUF repo — those are usually named "
            f"'<model>-GGUF', e.g. 'unsloth/Qwen3-8B-GGUF'. To run a safetensors model, "
            f"use a GGUF conversion of it, or set OLLAMA_MODEL to an Ollama library "
            f"model instead. Files in the repo: {sorted(files)[:15]}"
        )

    if model_file:
        wanted = os.path.basename(model_file.strip().lstrip("./")).lower()
        for candidate in ggufs:
            if candidate == model_file.strip() or os.path.basename(candidate).lower() == wanted:
                key = _shard_key(candidate)
                group = next(
                    g for g in group_ggufs(ggufs) if _shard_key(g[0]) == key and (
                        key is not None or g[0] == candidate
                    )
                )
                return validate_group(group, repo_id)
        raise ValueError(
            f"HF_MODEL_FILE '{model_file}' is not in '{repo_id}'. "
            f"Available GGUF files: {ggufs[:20]}"
        )

    # Explicit HF_MODEL_FILE above may name a projector deliberately; automatic
    # selection must never land on one.
    candidates = [f for f in ggufs if not NON_MODEL_GGUF_RE.search(os.path.basename(f))]
    skipped = [f for f in ggufs if f not in candidates]
    if skipped:
        print(
            f"Ignoring {len(skipped)} non-model GGUF file(s) in '{repo_id}': "
            f"{[os.path.basename(f) for f in skipped]}",
            flush=True,
        )
    if not candidates:
        raise ValueError(
            f"'{repo_id}' contains only non-model GGUF files "
            f"({[os.path.basename(f) for f in skipped]}). These are multimodal "
            f"projectors or similar sidecars, not language models, so Ollama cannot "
            f"serve them. Point HF_MODEL at a repo with model weights."
        )

    groups = group_ggufs(candidates)

    if quantization:
        pattern = _quant_pattern(quantization)
        matches = [g for g in groups if pattern.search(_stem(g[0]))]
        if len(matches) == 1:
            return validate_group(matches[0], repo_id)
        if not matches:
            raise ValueError(
                f"No GGUF in '{repo_id}' matches HF_QUANTIZATION='{quantization}'. "
                f"Available quantizations: {available_quants(candidates)}. "
                f"GGUF files: {[g[0] for g in groups][:20]}"
            )
        raise ValueError(
            f"HF_QUANTIZATION='{quantization}' is ambiguous in '{repo_id}' — it matches "
            f"{[g[0] for g in matches][:10]}. Set HF_MODEL_FILE to the exact filename."
        )

    if len(groups) == 1:
        return validate_group(groups[0], repo_id)

    def group_size(group):
        return sum((sizes or {}).get(path, 0) for path in group)

    # No quantization asked for. Prefer Q4_K_M, which is what Ollama's own puller
    # picks and what model cards assume; fall back to the smallest variant only
    # when the repo doesn't ship it.
    preferred_pattern = _quant_pattern(PREFERRED_QUANT)
    for group in groups:
        if preferred_pattern.search(_stem(group[0])):
            try:
                chosen = validate_group(group, repo_id)
            except ValueError:
                continue
            print(
                f"HF_QUANTIZATION not set — defaulting to {PREFERRED_QUANT} in "
                f"'{repo_id}': {os.path.basename(chosen[0])}"
                + (f" ({_human_size(group_size(group))})" if sizes else "")
                + f". Available quantizations: {available_quants(candidates)}",
                flush=True,
            )
            return chosen

    if sizes:
        for group in sorted(groups, key=lambda g: (group_size(g), g[0])):
            if group_size(group) <= 0:
                continue
            try:
                chosen = validate_group(group, repo_id)
            except ValueError:
                continue  # incomplete split GGUF — try the next size up
            print(
                f"HF_QUANTIZATION not set and no {PREFERRED_QUANT} in '{repo_id}' — "
                f"falling back to the smallest GGUF: {os.path.basename(chosen[0])} "
                f"({_human_size(group_size(group))}). Set HF_QUANTIZATION explicitly "
                f"if you need a higher-quality quantization. "
                f"Available quantizations: {available_quants(candidates)}",
                flush=True,
            )
            return chosen

    raise ValueError(
        f"'{repo_id}' contains {len(groups)} GGUF variants and their sizes could not be "
        f"determined, so the smallest can't be picked automatically — set "
        f"HF_QUANTIZATION to choose one. Available quantizations: {available_quants(ggufs)}."
    )


def acquire_gguf(repo_id, quantization, model_file):
    """Absolute path(s) of the GGUF to register, preferring Runpod's model store.

    The model store is the intended path: Runpod prefills it before the worker
    starts and doesn't bill for the download. Fetching from Hugging Face directly
    is the fallback, and says so loudly because it costs cold-start time.
    """
    snapshot = find_cached_snapshot(repo_id)
    if snapshot:
        available = list_snapshot_files(snapshot)
        sizes = {}
        for rel in available:
            if rel.lower().endswith(".gguf"):
                try:
                    sizes[rel] = os.path.getsize(os.path.realpath(os.path.join(snapshot, rel)))
                except OSError:
                    pass
        selected = select_gguf(available, quantization, model_file, repo_id, sizes)
        paths = [os.path.realpath(os.path.join(snapshot, rel)) for rel in selected]
        if all(os.path.isfile(p) for p in paths):
            print(f"[ModelStore] Using snapshot {snapshot}", flush=True)
            for path in paths:
                size = _human_size(os.path.getsize(path))
                print(f"[ModelStore]   {os.path.basename(path)} ({size})", flush=True)
            return paths
        print(
            f"[ModelStore] Snapshot {snapshot} is missing files for the selected "
            f"quantization — falling back to download",
            flush=True,
        )
    else:
        print(
            f"WARN: no cached snapshot for '{repo_id}' under {RUNPOD_MODEL_CACHE_DIR}.\n"
            f"      Downloading from Hugging Face instead (billed cold-start time).\n"
            f"      To use Runpod's model store, set the endpoint's Model field to "
            f"'{repo_id}'.",
            flush=True,
        )

    # Imported lazily so endpoints that only use OLLAMA_MODEL never pay for it.
    # list_repo_tree rather than list_repo_files: it returns sizes in the same
    # call, which the smallest-quantization default needs.
    from huggingface_hub import hf_hub_download, list_repo_tree

    token = HF_TOKEN or None
    entries = list(list_repo_tree(repo_id, recursive=True, token=token))
    download_root = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/hub"
    )
    available = [e.path for e in entries]
    sizes = {
        e.path: e.size
        for e in entries
        if e.path.lower().endswith(".gguf") and getattr(e, "size", None)
    }
    selected = select_gguf(available, quantization, model_file, repo_id, sizes)
    wanted = sum(sizes.get(rel, 0) for rel in selected)
    if wanted:
        require_free_space(download_root, wanted, f"downloading {repo_id}")

    paths = []
    for rel in selected:
        print(f"Downloading {repo_id}/{rel} ({_human_size(sizes.get(rel, 0))})", flush=True)
        paths.append(os.path.realpath(hf_hub_download(repo_id, rel, token=token)))
    return paths


def file_digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def blob_present(digest):
    response = session.head(f"{OLLAMA_BASE_URL}/api/blobs/{digest}", timeout=30)
    return response.status_code == 200


def link_blob(path, digest):
    """Hard-link a GGUF into Ollama's blob store instead of uploading it.

    Ollama resolves blobs with a bare stat and keeps no index, so a correctly
    named hard link is indistinguishable from an upload — and moves no bytes.
    Returns False when the file and the store are on different filesystems.
    """
    blobs_dir = os.path.join(OLLAMA_MODELS_DIR, "blobs")
    final = os.path.join(blobs_dir, digest.replace(":", "-"))
    # Unique per attempt: several cold workers can share one network volume, and a
    # fixed temp name lets them delete each other's in-flight file.
    temp = f"{final}.{os.getpid()}.{uuid.uuid4().hex[:8]}.link"
    try:
        os.makedirs(blobs_dir, exist_ok=True)
        if os.path.lexists(temp):
            os.remove(temp)
        os.link(path, temp)
        os.replace(temp, final)
        return True
    except OSError as err:
        print(f"Hard link into {blobs_dir} failed ({err}) — uploading blob instead", flush=True)
        try:
            os.remove(temp)
        except OSError:
            pass
        return False


def upload_blob(path, digest):
    with open(path, "rb") as f:
        response = session.post(f"{OLLAMA_BASE_URL}/api/blobs/{digest}", data=f, timeout=7200)
    response.raise_for_status()


def create_model_from_gguf(model, paths):
    """Register local GGUF file(s) with Ollama under `model`."""
    total = sum(os.path.getsize(p) for p in paths)
    files = {}
    for path in paths:
        digest = file_digest(path)
        linked = True
        if not blob_present(digest):
            linked = link_blob(path, digest)
            if not linked:
                # Cross-filesystem: the bytes get uploaded as well as re-written.
                require_free_space(OLLAMA_MODELS_DIR, total * 3, f"registering '{model}'")
                upload_blob(path, digest)
        files[os.path.basename(path)] = digest

    # Ollama writes a COPY temp plus the final blob, so ~2x on top of the link.
    require_free_space(OLLAMA_MODELS_DIR, total * 2, f"registering '{model}'")

    payload = {"model": model, "files": files, "stream": False}
    if OLLAMA_TEMPLATE:
        payload["template"] = OLLAMA_TEMPLATE
    print(f"Registering '{model}' from {sorted(files)}", flush=True)
    response = session.post(f"{OLLAMA_BASE_URL}/api/create", json=payload, timeout=7200)
    response.raise_for_status()

    # /api/create answers 200 and reports failures inside the body, so the status
    # code alone proves nothing.
    error = None
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("error"):
            error = event["error"]
    if error:
        if is_out_of_space(error):
            raise ValueError(
                f"Out of disk space registering '{model}': {error}. Ollama re-writes "
                f"the GGUF when importing it, so this needs roughly 3x the model size "
                f"({_human_size(total * 3)} for this model) at peak. Increase the "
                f"endpoint's container disk, or attach a network volume."
            )
        raise ValueError(f"ollama create failed for '{model}': {error}")

    describe_model(model)


def pull_hf_model_with_token(model):
    """Pull an hf.co model using an HF access token (gated/private repos).

    Ollama's native pull authenticates with an SSH key, which serverless
    workers can't practically register. Instead, fetch the same manifest and
    blobs from Hugging Face's Ollama-compatible registry with bearer auth and
    write them straight into Ollama's model store.
    """
    if not model.startswith("hf.co/"):
        raise ValueError(f"expected an 'hf.co/' model reference, got '{model}'")
    ref = model.split("/", 1)[1]  # strip "hf.co/"
    repo, _, tag = ref.partition(":")
    tag = tag or "latest"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    manifest_response = session.get(
        f"https://huggingface.co/v2/{repo}/manifests/{tag}",
        headers={**headers, "Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        timeout=60,
    )
    manifest_response.raise_for_status()
    manifest = manifest_response.json()

    blobs_dir = os.path.join(OLLAMA_MODELS_DIR, "blobs")
    manifest_dir = os.path.join(OLLAMA_MODELS_DIR, "manifests", "hf.co", repo)
    os.makedirs(blobs_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    for layer in [manifest["config"], *manifest["layers"]]:
        digest = layer["digest"]
        blob_path = os.path.join(blobs_dir, digest.replace(":", "-"))
        if os.path.exists(blob_path) and os.path.getsize(blob_path) == layer["size"]:
            continue
        hasher = hashlib.sha256()
        # Unique per attempt so concurrent workers sharing a volume don't clobber
        # each other's partial download.
        partial = f"{blob_path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial"
        with session.get(
            f"https://huggingface.co/v2/{repo}/blobs/{digest}",
            headers=headers,
            stream=True,
            timeout=3600,
        ) as blob_response:
            blob_response.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in blob_response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    hasher.update(chunk)
        if f"sha256:{hasher.hexdigest()}" != digest:
            os.remove(partial)
            raise ValueError(f"Digest mismatch downloading blob {digest} for {model}")
        os.replace(partial, blob_path)

    with open(os.path.join(manifest_dir, tag), "w") as f:
        json.dump(manifest, f)


def ensure_model(model):
    model = normalize_model_name(model)
    local = get_local_models()
    if model in local or f"{model}:latest" in local:
        return
    if HF_TOKEN and model.startswith("hf.co/"):
        pull_hf_model_with_token(model)
        return
    response = session.post(
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model, "stream": False},
        timeout=3600,
    )
    if not response.ok:
        raise ValueError(
            f"ollama pull failed for '{model}': {ollama_error(response)}. If this is a "
            f"Hugging Face repo id, set it as HF_MODEL on the endpoint instead of passing "
            f"it as 'model', or reference it as 'hf.co/<org>/<repo>:<quant>'."
        )


def ensure_default_model():
    """Prepare the model the endpoint is configured for.

    Idempotent, so start.sh can call it at boot and the handler can call it again
    on the first request if that startup attempt failed.
    """
    model = resolve_default_model()
    if not model:
        return ""
    local = get_local_models()
    if model in local or f"{model}:latest" in local:
        print(f"Model already present: {model}", flush=True)
        describe_model(model)
        return model
    if HF_MODEL:
        if HF_MODEL != HF_MODEL_RAW or HF_QUANTIZATION != HF_QUANTIZATION_RAW:
            print(
                f"Read HF_MODEL='{HF_MODEL_RAW}' as repo '{HF_MODEL}'"
                + (f", quantization '{HF_QUANTIZATION}'" if HF_QUANTIZATION else ""),
                flush=True,
            )
        create_model_from_gguf(model, acquire_gguf(HF_MODEL, HF_QUANTIZATION, HF_MODEL_FILE))
    else:
        ensure_model(model)
    return model


# Go-style durations, which is what Ollama's env vars use. "ms" before "m" so
# "500ms" doesn't half-match as "500m" plus a dangling "s".
_DURATION_RE = re.compile(r"^([+-]?)((?:\d+(?:\.\d+)?(?:ms|us|ns|h|m|s))+)$")
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|us|ns|h|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

# Fallback and cap for the warm-up bound: never let boot hang on a load forever,
# even when OLLAMA_LOAD_TIMEOUT is unset, unparseable, or "retry indefinitely".
_WARMUP_FALLBACK_S = 3600.0
# Client-side margin over Ollama's own load timeout, so the server's error — which
# says *why* the load failed — wins the race against a blank client timeout.
_WARMUP_MARGIN_S = 60.0


def _duration_seconds(value, default):
    """Parse a Go time.ParseDuration string ('60m', '1h30m', '90s') to seconds.

    Ollama parses OLLAMA_LOAD_TIMEOUT with Go's parser; mirroring it here keeps
    one env var meaning one thing. Unparseable values return `default` rather
    than raising — a bad env var must not break boot.
    """
    match = _DURATION_RE.match((value or "").strip().lower())
    if not match:
        return default
    total = sum(
        float(number) * _UNIT_SECONDS[unit]
        for number, unit in _DURATION_PART_RE.findall(match.group(2))
    )
    return -total if match.group(1) == "-" else total


def _warmup_timeout_seconds():
    """How long the boot warm-up may spend loading the model into memory.

    Ollama itself gives up after OLLAMA_LOAD_TIMEOUT (its default is 5m; this
    image's Hub default is 60m; non-positive means retry forever), so the client
    bound sits just above that: Ollama's load-timeout error names the model and
    the cause, which beats a mute client-side timeout. Non-positive or
    unparseable values are capped at an hour so warm-up is never unbounded.
    """
    timeout = _duration_seconds(os.environ.get("OLLAMA_LOAD_TIMEOUT"), _WARMUP_FALLBACK_S)
    if timeout <= 0:
        timeout = _WARMUP_FALLBACK_S
    return timeout + _WARMUP_MARGIN_S


def _report_residency(model):
    """Log where the loaded weights actually ended up (VRAM vs CPU spill).

    /api/ps is the only place Ollama reports size_vram; a partial-VRAM load is
    the "first request is mysteriously slow" case worth flagging at boot.
    """
    response = session.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=30)
    if not response.ok:
        return
    for entry in response.json().get("models", []):
        if entry.get("name") not in (model, f"{model}:latest"):
            continue
        size = entry.get("size") or 0
        vram = entry.get("size_vram") or 0
        if size and vram < size:
            print(
                f"WARN: '{model}' loaded, but only {_human_size(vram)} of "
                f"{_human_size(size)} fits in VRAM — the rest is offloaded to CPU and "
                f"responses will be much slower. Pick a GPU with more VRAM, a smaller "
                f"quantization via HF_QUANTIZATION, or a lower OLLAMA_CONTEXT_LENGTH.",
                flush=True,
            )
        else:
            print(f"'{model}' is resident in GPU memory ({_human_size(vram)})", flush=True)
        return
    print(f"WARN: '{model}' answered the warm-up but is not listed by /api/ps", flush=True)


def warm_model(model):
    """Load `model`'s weights into GPU memory before any user request arrives.

    Registering a model (ensure_default_model) only writes it to disk — Ollama
    loads weights into VRAM lazily, on the first inference. Without this, the
    first user request pays the whole multi-minute load and can blow the
    endpoint's execution timeout. An empty prompt is Ollama's documented
    "just load it" request: no tokens are generated, and the image's
    OLLAMA_KEEP_ALIVE=-1 default keeps the weights resident afterwards.
    """
    timeout = _warmup_timeout_seconds()
    print(
        f"Loading '{model}' into GPU memory (bounded at {int(timeout)}s "
        f"by OLLAMA_LOAD_TIMEOUT)...",
        flush=True,
    )
    started = time.monotonic()
    response = session.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": "", "stream": False},
        timeout=timeout,
    )
    error = ollama_error(response)
    if error:
        raise ValueError(error)
    print(f"Loaded '{model}' in {time.monotonic() - started:.1f}s", flush=True)
    try:
        _report_residency(model)
    except (requests.RequestException, ValueError) as err:
        print(f"WARN: could not read /api/ps after warming '{model}': {err}", flush=True)


def warm_default_model():
    """Best-effort boot-time warm-up of the model the endpoint is configured for.

    Deliberately non-fatal: the model is already registered, so the handler still
    works if this fails — the first request just pays the load, as it did before
    warm-up existed. But say exactly what failed, so "too big for this GPU" or
    "load timed out" is diagnosable from the boot log instead of from a timed-out
    first request. Returns True when the model is warm.
    """
    model = resolve_default_model()
    if not model:
        return False
    try:
        warm_model(model)
        return True
    except (requests.RequestException, ValueError, OSError) as err:
        print(
            f"WARN: could not pre-load '{model}' into GPU memory — the first request "
            f"will trigger the load instead and may be slow or time out. Cause: {err}",
            flush=True,
        )
        return False


def handler(job):
    job_input = job.get("input") or {}

    requested = job_input.get("model")
    model = normalize_model_name(requested) if requested else resolve_default_model()
    if not model:
        yield {
            "error": (
                "The 'model' value in the request input is blank. Omit it to use the "
                "model the endpoint is configured with, or pass a valid model name."
            )
        }
        return

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if not messages and not prompt:
        yield {"error": "Provide either 'messages' (chat) or 'prompt' (completion) in input."}
        return

    try:
        if requested:
            ensure_model(model)
        else:
            ensure_default_model()
    except (requests.RequestException, ValueError, OSError) as err:
        yield {"error": f"Failed to prepare model '{model}': {err}"}
        return

    if messages:
        endpoint = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {"model": model, "messages": messages}
    else:
        endpoint = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {"model": model, "prompt": prompt}

    for key in ("options", "format", "keep_alive", "tools", "system", "template"):
        if key in job_input:
            payload[key] = job_input[key]

    stream = bool(job_input.get("stream", False))
    payload["stream"] = stream

    try:
        if stream:
            with session.post(endpoint, json=payload, stream=True, timeout=3600) as response:
                if not response.ok:
                    yield {"error": f"Ollama request failed: {ollama_error(response)}"}
                    return
                for line in response.iter_lines():
                    if not line:
                        continue
                    yield line.decode("utf-8")
        else:
            response = session.post(endpoint, json=payload, timeout=3600)
            if not response.ok:
                yield {"error": f"Ollama request failed: {ollama_error(response)}"}
                return
            yield response.json()
    except requests.RequestException as err:
        yield {"error": f"Ollama request failed: {err}"}


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )
