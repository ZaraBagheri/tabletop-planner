"""Train/save/reload a tiny RANDOM Qwen-shaped model offline. Not a capable planner."""
import json
from pathlib import Path
import tempfile
import signal
from unittest.mock import patch


def run(root):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from tabletop_planner.data import generate_dataset
    from tabletop_planner.training import train, encode_example, collate
    from tabletop_planner.planners import ModelPlanner

    root = Path(root)
    model_dir, data = root/"tiny_random_base", root/"data"
    model_dir.mkdir(parents=True)
    generate_dataset(data, 10, 5, 5)
    rows = [json.loads(x) for x in (data/"train.jsonl").read_text().splitlines()]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator([m["content"] for r in rows for m in r["messages"]],
                                   WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]", "[SYS]", "[USR]", "[AST]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    fast.chat_template = "{% for m in messages %}{{ {'system':'[SYS]', 'user':'[USR]', 'assistant':'[AST]'}[m['role']] }}{{ m['content'] }}[EOS]{% endfor %}{% if add_generation_prompt %}[AST]{% endif %}"
    fast.save_pretrained(model_dir)
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                        max_position_embeddings=2048, bos_token_id=fast.eos_token_id,
                                        eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id))
    model.save_pretrained(model_dir, safe_serialization=True)
    example = encode_example(fast, rows[0], 1536)
    prefix = fast.apply_chat_template(rows[0]["messages"][:-1], add_generation_prompt=True)
    assert all(x == -100 for x in example["labels"][:len(prefix)])
    assert any(x != -100 for x in example["labels"])
    batch = collate([example, encode_example(fast, rows[-1], 1536)], fast.pad_token_id)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    output = root/"adapter"
    report = train(data, output, str(model_dir), epochs=1, accumulation=2, max_steps=2, device="cpu", checkpoint_every=1)
    from safetensors.torch import load_file
    state = load_file(output/"adapter_model.safetensors")
    assert any(torch.count_nonzero(v) for k, v in state.items() if "lora_B" in k)
    loaded = ModelPlanner(str(model_dir), str(output), device="cpu")
    assert loaded.name == "finetuned"
    assert report["optimizer_steps"] == 2
    assert loaded.observation_format == "semantic-v2"
    assert (output/"checkpoints/step-000001/training_state.pt").exists()
    resumed, uninterrupted = root/"resumed", root/"uninterrupted"
    continued = train(data, resumed, str(model_dir), epochs=1, max_steps=4, resume=output/"checkpoints/step-000002", device="cpu")
    train(data, uninterrupted, str(model_dir), epochs=1, accumulation=2, max_steps=4, device="cpu")
    resumed_weights, full_weights = load_file(resumed/"adapter_model.safetensors"), load_file(uninterrupted/"adapter_model.safetensors")
    assert continued["optimizer_steps"] == 4
    assert all(torch.equal(resumed_weights[key], full_weights[key]) for key in full_weights), "Resumed weights differ from uninterrupted training"
    interrupted = root/"interrupted"
    original_step = torch.optim.AdamW.step
    def interrupt_after_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        signal.raise_signal(signal.SIGINT)
        return result
    with patch.object(torch.optim.AdamW, "step", interrupt_after_step):
        stopped = train(data, interrupted, str(model_dir), epochs=1, accumulation=2, max_steps=4, device="cpu")
    assert stopped["stop_reason"] == "interrupted" and stopped["optimizer_steps"] == 1
    assert (interrupted/"training_state.pt").exists()
    recovered = root/"recovered_after_interrupt"
    train(data, recovered, resume=interrupted, epochs=1, max_steps=4, device="cpu")
    recovered_weights = load_file(recovered/"adapter_model.safetensors")
    assert all(torch.equal(recovered_weights[key], full_weights[key]) for key in full_weights), "Interrupt/resume changed final weights"
    # A periodic checkpoint at the last batch precedes epoch validation/advancement.
    boundary_steps = (len(rows)+1)//2
    boundary, boundary_resumed, boundary_full = root/"boundary", root/"boundary_resumed", root/"boundary_full"
    train(data, boundary, str(model_dir), epochs=2, accumulation=2, max_steps=boundary_steps,
          checkpoint_every=boundary_steps, device="cpu")
    train(data, boundary_resumed, resume=boundary/"checkpoints"/f"step-{boundary_steps:06d}",
          epochs=2, max_steps=boundary_steps+2, device="cpu")
    train(data, boundary_full, str(model_dir), epochs=2, accumulation=2,
          max_steps=boundary_steps+2, device="cpu")
    a, b = load_file(boundary_resumed/"adapter_model.safetensors"), load_file(boundary_full/"adapter_model.safetensors")
    assert all(torch.equal(a[key], b[key]) for key in b), "Epoch-boundary resume changed final weights"
    bad_data = root/"bad_data"
    bad_data.mkdir()
    for split in ("train", "val"):
        (bad_data/f"{split}.jsonl").write_bytes((data/f"{split}.jsonl").read_bytes()+b"\n")
    try:
        train(bad_data, root/"bad_resume", resume=output, epochs=1, max_steps=4, device="cpu")
    except ValueError as exc:
        assert "hashes" in str(exc)
    else:
        raise AssertionError("Changed resume data was accepted")
    print("PASS: masks, trained weights, save/reload, exact CPU resume within/across epochs, checkpoints, graceful interrupt/resume, and changed-data rejection")
    return report


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        run(directory)
