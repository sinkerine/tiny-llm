from .qwen3_week1 import Qwen3ModelWeek1


def dispatch_model(model_name: str, hf_model, week: int, **kwargs):
    if week != 1:
        raise ValueError("The CUDA runtime currently supports Week 1 only")
    if hf_model.config.model_type != "qwen3":
        raise ValueError(f"Unsupported model: {model_name}")
    return Qwen3ModelWeek1(hf_model, **kwargs)
