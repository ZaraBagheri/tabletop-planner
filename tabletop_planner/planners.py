from .observations import messages_for, load_planner_config
from .domain import Action


def choose_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelPlanner:
    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", adapter=None,
                 device="auto", revision="main", max_length=1536, observation_format="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = choose_device(torch, device)
        self.name = "finetuned" if adapter else "base"
        self.max_length = max_length
        self.input_config = load_planner_config(adapter, observation_format)
        self.observation_format = self.input_config["observation_format"]
        if adapter:
            from peft import PeftConfig
            config = PeftConfig.from_pretrained(adapter)
            if config.base_model_name_or_path != model:
                raise ValueError("Adapter base model differs from --model; supply the recorded base model")
            revision = config.revision or revision
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model, revision=None if adapter else revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision,
                                                        torch_dtype=dtype, use_safetensors=True)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(self.device).eval()
        self.last_text = ""

    def next_action(self, instruction, observation):
        prompt = self.tokenizer.apply_chat_template(messages_for(instruction, observation,
            self.observation_format, self.input_config["system_prompt"]),
                                                    tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        if inputs.input_ids.shape[1] > self.max_length:
            raise ValueError("Prompt exceeds the configured context budget")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=96, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id)
        self.last_text = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return Action.parse(self.last_text)
