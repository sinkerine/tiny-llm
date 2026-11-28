from datetime import datetime
from typing import Any

import torch

from .kv_cache import BatchingKvCache


class _HFDetokenizer:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.tokens: list[int] = []

    def reset(self) -> None:
        self.tokens = []

    def add_token(self, token: int) -> None:
        self.tokens.append(int(token))

    def finalize(self) -> None:
        return None

    @property
    def text(self) -> str:
        return self.tokenizer.decode(self.tokens)


def _make_detokenizer(tokenizer: Any) -> Any:
    if hasattr(tokenizer, "detokenizer") and hasattr(tokenizer, "_tokenizer"):
        return tokenizer.detokenizer.__class__(tokenizer._tokenizer)
    return _HFDetokenizer(tokenizer)


def _model_device(model: Any) -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _step(model, y, offsets, kv_cache):
    logits = model(y, offsets, kv_cache, logits_to_keep=1)
    logits = logits[:, -1, :].float()
    logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.argmax(logprobs, dim=-1)


class Request:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
        max_seq_len: int | None = None,
    ):
        self.prompt = prompt
        self.model = model
        self.device = _model_device(model)
        self.detokenizer = _make_detokenizer(tokenizer)
        self.prefill_tokens = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
            device=self.device,
        )
        if max_seq_len is not None and self.prefill_tokens.numel() > max_seq_len:
            raise ValueError(
                f"Prompt has {self.prefill_tokens.numel()} tokens, which exceeds "
                f"max_seq_len={max_seq_len}"
            )
        self.kv_cache = model.create_kv_cache()
        self.prefill_max_step = prefill_max_step
        self.max_seq_len = max_seq_len
        self.is_done = False
        self.is_prefill_done = False
        self.finish_reason = None
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

    def try_prefill(self):
        """
        Prefill this request up to max_step size, returns None if prefill is not done
        """
        if self.is_prefill_done:
            raise ValueError("prefill called after done")
        tokens_to_prefill = min(
            self.prefill_max_step, self.prefill_tokens.numel() - self.offset
        )
        token = _step(
            self.model,
            self.prefill_tokens[self.offset : self.offset + tokens_to_prefill][None],
            [self.offset],
            self.kv_cache,
        )
        self.offset += tokens_to_prefill

        for layer_cache in self.kv_cache:
            layer_cache.materialize()
        if self.offset == self.prefill_tokens.numel():
            self.is_prefill_done = True
            if self.max_seq_len is not None and self.offset >= self.max_seq_len:
                self.is_done = True
                self.finish_reason = "max seq len"
            else:
                self.decode_done(int(token.item()), False)

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        if token == self.eos_token_id:
            self.is_done = True
            self.finish_reason = "EOS"
            return
        self.detokenizer.add_token(token)
        self.next_token = token
        if update_offset:
            self.offset += 1

    def text(self):
        return self.detokenizer.text

    def reaches_max_seq_len(self, max_seq_len: int) -> bool:
        return self.next_token is not None and self.offset + 1 >= max_seq_len


def _print_progress(
    requests: list[Request | None],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    print(f"  --- {datetime.now() - start_time}")
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]
    for i, request in enumerate(requests):
        if request is None:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = request.text()[-80:].replace("\n", " ")
            print(
                f"{animation_frame} Decode [req {request.prompt_idx}, {request.offset}]: {text_preview}",
                flush=True,
            )
    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        percentage = (
            pending_prefill_request.offset
            / pending_prefill_request.prefill_tokens.numel()
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {percentage:.2f}% ({pending_prefill_request.prefill_tokens.numel() - pending_prefill_request.offset} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


@torch.inference_mode()
def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_seq_len: int = 512,
    batch_size: int = 5,
    prefill_step: int = 128,
):
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prefill_step <= 0:
        raise ValueError("prefill_step must be positive")

    device = _model_device(model)
    prompts = list(prompts)
    decode_requests: list[Request | None] = [None] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_request = None
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    try:
        while True:
            if (
                len(prompts) == 0
                and pending_prefill_request is None
                and all(req is None for req in decode_requests)
            ):
                break
            if len(prompts) > 0 and pending_prefill_request is None:
                prompt = prompts.pop(0)
                pending_prefill_request = Request(
                    model,
                    tokenizer,
                    prompt,
                    prefill_step,
                    next_request_idx,
                    max_seq_len=max_seq_len,
                )
                next_request_idx += 1

            if pending_prefill_request is not None:
                made_progress = False
                if not pending_prefill_request.is_prefill_done:
                    pending_prefill_request.try_prefill()
                    made_progress = True
                if pending_prefill_request.is_prefill_done:
                    request_finished = (
                        pending_prefill_request.is_done
                        or pending_prefill_request.reaches_max_seq_len(max_seq_len)
                    )
                    if request_finished:
                        text = pending_prefill_request.text()
                        for request_cache in pending_prefill_request.kv_cache:
                            request_cache.release()
                        result.append((pending_prefill_request.prompt_idx, text))
                        pending_prefill_request = None
                        made_progress = True
                    else:
                        prefill_kv_cache = pending_prefill_request.kv_cache
                        found_slot = False
                        for i in range(batch_size):
                            if decode_requests[i] is None:
                                for prefill_cache, batch_cache in zip(
                                    prefill_kv_cache, kv_cache
                                ):
                                    batch_cache.add_request(prefill_cache, i)
                                decode_requests[i] = pending_prefill_request
                                found_slot = True
                                made_progress = True
                                break
                        if found_slot:
                            pending_prefill_request = None
                if made_progress:
                    _print_progress(
                        decode_requests,
                        pending_prefill_request,
                        len(prompts),
                        progress_cnt,
                        start_time,
                    )
                    progress_cnt += 1

            if any(req is not None for req in decode_requests):
                next_tokens = []
                offsets = []
                for req in decode_requests:
                    if req is not None:
                        next_tokens.append(req.next_token)
                        offsets.append(req.offset)
                    else:
                        next_tokens.append(0)
                        offsets.append(0)
                next_tokens_t = torch.tensor(
                    next_tokens, dtype=torch.long, device=device
                ).reshape(-1, 1)
                next_tokens_t = _step(model, next_tokens_t, offsets, kv_cache)
                for i in range(batch_size):
                    req = decode_requests[i]
                    if req is not None:
                        req.decode_done(int(next_tokens_t[i].item()))
                        remove_reason = None
                        if req.is_done:
                            remove_reason = req.finish_reason
                        elif req.reaches_max_seq_len(max_seq_len):
                            remove_reason = "max seq len"
                        if remove_reason is not None:
                            print(
                                f"Removing request {i} due to {remove_reason}",
                                flush=True,
                            )
                            text = req.text()
                            for layer_cache in kv_cache:
                                layer_cache.remove_request(i)
                            result.append((req.prompt_idx, text))
                            decode_requests[i] = None
                _print_progress(
                    decode_requests,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1
    finally:
        live_caches = {}
        if pending_prefill_request is not None:
            for request_cache in pending_prefill_request.kv_cache:
                live_caches[id(request_cache)] = request_cache
        for batch_cache in kv_cache:
            for request_cache in batch_cache.kv_caches:
                if request_cache is not None:
                    live_caches[id(request_cache)] = request_cache
            batch_cache.kv_caches = [None] * batch_cache.max_active_requests
        for request_cache in live_caches.values():
            request_cache.release()
    return result
