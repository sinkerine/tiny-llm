from model_names import shortcut_name_to_full_name

from .qwen3_week1 import Qwen3ModelWeek1


def dispatch_model(model_name: str, hf_model, week: int, **kwargs):
    full_name = shortcut_name_to_full_name(model_name)
    model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
    is_qwen3 = full_name.startswith("Qwen/Qwen3") or model_type in (
        "qwen3",
        "qwen3_moe",
    )
    if week == 1 and is_qwen3 and model_type != "qwen3_moe":
        return Qwen3ModelWeek1(hf_model, **kwargs)
    if week == 2 and is_qwen3:
        from .qwen3_week2 import Qwen3ModelWeek2

        return Qwen3ModelWeek2(hf_model, **kwargs)
    if week == 3 and is_qwen3:
        from .qwen3_week3 import Qwen3ModelWeek3

        return Qwen3ModelWeek3(hf_model, **kwargs)
    raise ValueError(f"Unsupported model {model_name} for week {week}")


def dispatch_week3_batch_model(model_name: str, hf_model):
    """Build the dense-cache Week 3 scheduler model."""
    return dispatch_model(
        model_name,
        hf_model,
        week=2,
    )
