"""Completion-only LoRA training with resumable optimizer-boundary checkpoints."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import signal
import tempfile
import threading
import time


def encode_example(tokenizer, row, max_length):
    messages = row["messages"]
    if len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected exactly system, user, assistant training messages")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("Tokenizer chat template does not give a stable assistant prefix")
    if len(full) > max_length:
        raise ValueError(f"Example has {len(full)} tokens > {max_length}; increase --max-length (no silent truncation)")
    if len(full) <= len(prefix):
        raise ValueError("No completion tokens in training example")
    return {"input_ids": full, "attention_mask": [1]*len(full),
            "labels": [-100]*len(prefix)+full[len(prefix):]}


def collate(batch, pad_id):
    import torch
    length = max(len(x["input_ids"]) for x in batch)
    return {key: torch.tensor([x[key]+[pad]*(length-len(x[key])) for x in batch], dtype=torch.long)
            for key, pad in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))}


def train(data_dir, output, model="Qwen/Qwen2.5-0.5B-Instruct", revision="main", epochs=3,
          batch_size=1, accumulation=8, lr=2e-4, rank=8, max_length=1536,
          device="auto", max_steps=None, seed=42, *, resume=None, checkpoint_every=25, log_every=5):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from .observations import config_from_rows
    from .planners import choose_device

    output, data_dir = Path(output), Path(data_dir)
    state = None
    if resume:
        resume = Path(resume)
        if not (resume/"training_state.pt").exists():
            raise ValueError("Resume requires training_state.pt; older adapter-only runs cannot be resumed")
        state = torch.load(resume/"training_state.pt", map_location="cpu", weights_only=True)
        if state.get("version") != 1:
            raise ValueError("Unsupported training checkpoint version")
        saved = state["configuration"]
        model, revision = saved["model"], saved["revision"]
        batch_size, accumulation, lr = saved["batch_size"], saved["accumulation"], saved["lr"]
        rank, max_length, seed = saved["rank"], saved["max_length"], saved["seed"]
    if min(epochs, batch_size, accumulation, rank, max_length, checkpoint_every, log_every) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Training sizes, checkpoint/log intervals, and learning rate must be positive")
    if max_steps is not None and (max_steps < 1 or state and max_steps <= state["steps"]):
        raise ValueError("max_steps must exceed the saved optimizer step count")
    if output.exists() and any(output.iterdir()):
        if not state or state["run_output"] != str(output.resolve()):
            raise ValueError("Use a new/empty output directory, or resume into the same run directory")
        report_path = output/"training_report.json"
        if report_path.exists() and json.loads(report_path.read_text())["optimizer_steps"] > state["steps"]:
            raise ValueError("A newer result exists here; resume the older checkpoint into a new output directory")
        newer = [path for path in (output/"checkpoints").glob("step-*")
                 if path.is_dir() and path.name[5:].isdigit() and int(path.name[5:]) > state["steps"]]
        if newer:
            raise ValueError("Newer checkpoints exist here; resume the older checkpoint into a new output directory")
    raw, rows = {}, {}
    for split in ("train", "val"):
        raw[split] = (data_dir/f"{split}.jsonl").read_bytes()
        rows[split] = [json.loads(line) for line in raw[split].decode().splitlines() if line.strip()]
        if not rows[split]:
            raise ValueError(f"Empty {split} split")
    if {r["episode_id"] for r in rows["train"]} & {r["episode_id"] for r in rows["val"]}:
        raise ValueError("Train/validation episode leakage")
    hashes = {split: hashlib.sha256(value).hexdigest() for split, value in raw.items()}
    inputs = config_from_rows(rows["train"] + rows["val"])
    if state and (state["data_sha256"] != hashes or state["input_config"] != inputs):
        raise ValueError("Resume dataset hashes or model input format differ from the saved checkpoint")
    if state and epochs <= state["epoch"]:
        raise ValueError("Increase --epochs to continue beyond the completed epochs")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device(torch, device)
    print(f"device={device} train_examples={len(rows['train'])} validation_examples={len(rows['val'])} observation_format={inputs['observation_format']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(resume or model, revision=None if resume else revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = {split: [encode_example(tokenizer, row, max_length) for row in values] for split, values in rows.items()}
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=dtype, use_safetensors=True)
    resolved_revision = getattr(base.config, "_commit_hash", None) or revision
    base.config.use_cache = False
    if resume:
        net = PeftModel.from_pretrained(base, resume, is_trainable=True)
    else:
        net = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", revision=resolved_revision,
                            r=rank, lora_alpha=2*rank, lora_dropout=.05, target_modules=["q_proj", "v_proj"]))
    net.to(device)
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    net.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    configuration = {"model": model, "revision": resolved_revision, "batch_size": batch_size,
                     "accumulation": accumulation, "lr": lr, "rank": rank, "max_length": max_length, "seed": seed}
    progress = {"steps": 0, "epoch": 0, "next_batch": 0, "epoch_loss_sum": 0., "epoch_batches": 0,
                "history": [], "validation_loss_before": None, "elapsed_seconds": 0.}
    if state:
        optimizer.load_state_dict(state["optimizer"])
        progress.update(state["progress"])
        random.setstate(state["random_state"])
        torch.set_rng_state(state["torch_rng"])
        if device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if device == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        print(f"resumed step={progress['steps']} epoch={progress['epoch']+1} next_batch={progress['next_batch']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    start, initial_steps = time.monotonic(), progress["steps"]
    prior_elapsed = progress["elapsed_seconds"]
    stop_requested = False
    previous_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            print("Stop requested: finishing the current optimizer update, then saving. Please wait.", flush=True)
        stop_requested = True

    def elapsed():
        return prior_elapsed + time.monotonic()-start

    def validation_loss():
        net.eval()
        weighted, count = 0., 0
        loader = DataLoader(encoded["val"], batch_size=batch_size,
                            generator=torch.Generator().manual_seed(seed),
                            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
        print(f"validation batches=0/{len(loader)} elapsed_s={elapsed():.1f}", flush=True)
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if stop_requested:
                    print("Validation stopped before completion; partial loss is not reported.", flush=True)
                    return None
                batch = {key: value.to(device) for key, value in batch.items()}
                n = int((batch["labels"][:, 1:] != -100).sum())
                loss = net(**batch).loss.item()
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss")
                weighted += loss*n
                count += n
                if (index+1) % 10 == 0 or index+1 == len(loader):
                    print(f"validation batches={index+1}/{len(loader)} elapsed_s={elapsed():.1f}", flush=True)
        return weighted/count

    def report(reason):
        import transformers, peft
        return {"base_model": model, "requested_revision": revision, "resolved_revision": resolved_revision,
                "device": device, "seed": seed, "epochs_requested": epochs, "optimizer_steps": progress["steps"],
                "batch_size": batch_size, "gradient_accumulation": accumulation, "learning_rate": lr,
                "lora_rank": rank, "max_length": max_length,
                "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in net.parameters()),
                "validation_loss_before": progress["validation_loss_before"], "history": progress["history"],
                "elapsed_seconds": round(elapsed(), 3), "stop_reason": reason,
                "resumed_from": str(resume) if resume else None, "data_sha256": hashes,
                "observation_format": inputs["observation_format"], "prompt_sha256": inputs["prompt_sha256"],
                "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__, "peft": peft.__version__},
                "note": "Loss is not task success. Evaluate held-out symbolic and physics episodes."}

    def save(directory, reason):
        directory.mkdir(parents=True, exist_ok=True)
        net.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        (directory/"planner_config.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        (directory/"training_report.json").write_text(json.dumps(report(reason), indent=2), encoding="utf-8")
        saved_progress = dict(progress, elapsed_seconds=elapsed())
        snapshot = {"version": 1, "run_output": str(output.resolve()), "configuration": configuration,
                    "steps": progress["steps"], "epoch": progress["epoch"], "progress": saved_progress,
                    "data_sha256": hashes, "input_config": inputs, "optimizer": optimizer.state_dict(),
                    "random_state": random.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                    "mps_rng": torch.mps.get_rng_state() if device == "mps" else None}
        temporary_state = directory/"training_state.tmp"
        torch.save(snapshot, temporary_state)
        temporary_state.replace(directory/"training_state.pt")

    def checkpoint():
        parent = output/"checkpoints"
        parent.mkdir(exist_ok=True)
        destination = parent/f"step-{progress['steps']:06d}"
        if destination.exists():
            return
        temporary = Path(tempfile.mkdtemp(prefix=".saving-", dir=parent))
        try:
            save(temporary, "checkpoint")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        pointer = output/"latest_checkpoint.tmp"
        pointer.write_text(json.dumps({"path": str(destination.resolve()), "optimizer_steps": progress["steps"]}), encoding="utf-8")
        pointer.replace(output/"latest_checkpoint.json")
        print(f"checkpoint saved: {destination}", flush=True)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        if progress["validation_loss_before"] is None:
            progress["validation_loss_before"] = validation_loss()
        number_batches = math.ceil(len(encoded["train"])/batch_size)
        reached_limit = False
        while progress["epoch"] < epochs and not stop_requested and not reached_limit:
            epoch = progress["epoch"]
            order = torch.randperm(len(encoded["train"]), generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            first_batch = progress["next_batch"]
            remaining = [encoded["train"][index] for index in order[first_batch*batch_size:]]
            loader = DataLoader(remaining, batch_size=batch_size,
                                generator=torch.Generator().manual_seed(seed+epoch),
                                collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
            net.train()
            optimizer.zero_grad()
            for index, batch in enumerate(loader, start=first_batch):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = net(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; resume from the last complete checkpoint")
                progress["epoch_loss_sum"] += loss.item()
                progress["epoch_batches"] += 1
                group_start = (index//accumulation)*accumulation
                group_size = min(accumulation, number_batches-group_start)
                (loss/group_size).backward()
                if (index+1) % accumulation == 0 or index+1 == number_batches:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    progress["steps"] += 1
                    progress["next_batch"] = index+1
                    if progress["steps"] % log_every == 0:
                        print(f"epoch={epoch+1}/{epochs} batch={index+1}/{number_batches} optimizer_step={progress['steps']} loss={loss.item():.4f} elapsed_s={elapsed():.1f}", flush=True)
                    if progress["steps"] % checkpoint_every == 0:
                        checkpoint()
                    reached_limit = max_steps is not None and progress["steps"] >= max_steps
                    if stop_requested or reached_limit:
                        break
            completed_epoch = progress["next_batch"] == number_batches
            validation = None if stop_requested else validation_loss()
            progress["history"].append({"epoch": epoch+1, "epoch_completed": completed_epoch,
                "optimizer_steps": progress["steps"],
                "mean_batch_train_loss": progress["epoch_loss_sum"]/max(1, progress["epoch_batches"]),
                "validation_loss": validation})
            print(progress["history"][-1], flush=True)
            if completed_epoch:
                progress.update(epoch=epoch+1, next_batch=0, epoch_loss_sum=0., epoch_batches=0)
        reason = "interrupted" if stop_requested else "max_steps" if reached_limit else "completed"
        checkpoint()
        save(output, reason)
        final_report = report(reason)
        print(f"Training {reason}: saved step={progress['steps']} to {output}", flush=True)
        return final_report
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
