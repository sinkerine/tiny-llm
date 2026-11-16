from .qwen3_week1 import Qwen3ModelWeek1


def dispatch_model(model_name: str, hf_model, week: int, **kwargs):
    if week == 2:
        from .qwen3_week2 import Qwen3ModelWeek2

        return Qwen3ModelWeek2(hf_model, **kwargs)
    if week != 1:
        raise ValueError("The CUDA runtime supports Weeks 1 and 2")
    if hf_model.config.model_type != "qwen3":
        raise ValueError(f"Unsupported model: {model_name}")
    return Qwen3ModelWeek1(hf_model, **kwargs)
