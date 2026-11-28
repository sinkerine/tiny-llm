from typing import Any, Callable

import torch
from transformers import PreTrainedTokenizerBase, TextStreamer


def _validate_max_tokens(max_tokens: int) -> None:
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a non-negative integer")


def _release_kv_cache(kv_cache) -> None:
    if kv_cache is None:
        return
    for layer in kv_cache:
        release = getattr(layer, "release", None)
        if callable(release):
            release()


class _SimpleDetokenizer:
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


@torch.inference_mode()
def simple_generate(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
    max_tokens: int = 256,
) -> None:
    _validate_max_tokens(max_tokens)
    if max_tokens == 0:
        return
    tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        dtype=torch.long,
        device=getattr(model, "device", "cpu"),
    )
    if tokens.numel() == 0:
        raise ValueError("prompt must encode to at least one token")
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    for _ in range(max_tokens):
        logits = model(tokens[None])[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token = logprobs.argmax(dim=-1) if sampler is None else sampler(logprobs)
        if token.item() == tokenizer.eos_token_id:
            break
        tokens = torch.cat((tokens, token.to(device=tokens.device, dtype=torch.long)))
        streamer.put(token.cpu())
    streamer.end()


@torch.inference_mode()
def speculative_generate(
    draft_model: Any,
    model: Any,
    draft_tokenizer: Any,
    tokenizer: Any,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    if (
        not isinstance(proposal_length, int)
        or isinstance(proposal_length, bool)
        or proposal_length < 0
    ):
        raise ValueError("proposal_length must be a non-negative integer")
    _validate_max_tokens(max_tokens)
    if max_tokens == 0:
        return ""

    def _encode(tok):
        return [int(token) for token in tok.encode(prompt, add_special_tokens=False)]

    def _eos_ids(tok):
        eos_ids = getattr(tok, "eos_token_ids", None)
        if eos_ids is None:
            eos_ids = {tok.eos_token_id}
        return {int(token) for token in eos_ids}

    target_prompt_tokens = _encode(tokenizer)
    draft_prompt_tokens = _encode(draft_tokenizer)
    if not target_prompt_tokens:
        raise ValueError("prompt must encode to at least one token")
    if target_prompt_tokens != draft_prompt_tokens:
        raise ValueError("draft and target tokenizers encode the prompt differently")
    if _eos_ids(tokenizer) != _eos_ids(draft_tokenizer):
        raise ValueError("draft and target tokenizers use different EOS token ids")

    target_get_vocab = getattr(tokenizer, "get_vocab", None)
    draft_get_vocab = getattr(draft_tokenizer, "get_vocab", None)
    if not callable(target_get_vocab) or not callable(draft_get_vocab):
        raise ValueError(
            "draft and target tokenizers must expose comparable vocabularies"
        )
    if target_get_vocab() != draft_get_vocab():
        raise ValueError("draft and target tokenizers use different token ids")

    target_eos_ids = _eos_ids(tokenizer)
    draft_eos_ids = _eos_ids(draft_tokenizer)
    detokenizer = getattr(tokenizer, "detokenizer", None)
    if detokenizer is None:
        detokenizer = _SimpleDetokenizer(tokenizer)
    detokenizer.reset()
    emitted = 0

    target_device = getattr(model, "device", "cpu")
    draft_device = getattr(draft_model, "device", target_device)

    kv_cache = model.create_kv_cache()
    draft_kv_cache = None

    def _step(m, y, offset, cache, n_tokens=1):
        logits = m(y[None], offset, cache, logits_to_keep=n_tokens)
        if n_tokens > 1:
            logits = logits[:, -n_tokens:, :]
        else:
            logits = logits[:, -1, :]
        logits_f = logits.float()
        logprobs = logits_f - torch.logsumexp(logits_f, dim=-1, keepdim=True)
        y_out = torch.argmax(logprobs, dim=-1).to(torch.int32)
        return y_out, logprobs.squeeze(0)

    def _token_tensor(tokens, device):
        return torch.tensor(tokens, dtype=torch.int32, device=device)

    def _token_id(token):
        return int(token.item())

    def _prefill(m, prefill_tokens, cache, device):
        token, _ = _step(m, _token_tensor(prefill_tokens, device), 0, cache)
        return _token_id(token), len(prefill_tokens)

    def _rewind_cache(cache, revert_len):
        if revert_len == 0:
            return
        for layer in cache:
            layer.rewind(revert_len)

    def _assert_cache_offset(cache, expected):
        for layer in cache:
            if hasattr(layer, "offset"):
                assert layer.offset == expected

    def _print_text(text, progress):
        newline = "\n"
        print(f"+{progress} {text.replace(newline, ' ')[-80:]}")

    def _emit(token_ids):
        nonlocal emitted
        if emitted + len(token_ids) > max_tokens:
            raise AssertionError("speculative output exceeded max_tokens")
        for token_id in token_ids:
            detokenizer.add_token(token_id)
        emitted += len(token_ids)
        if token_ids:
            _print_text(detokenizer.text, len(token_ids))

    def _finish():
        finalize = getattr(detokenizer, "finalize", None)
        if callable(finalize):
            finalize()
        text = detokenizer.text
        print(text)
        return text

    def _target_only(token_id, offset):
        while emitted < max_tokens:
            if token_id in target_eos_ids:
                return _finish()
            _emit([token_id])
            if emitted == max_tokens:
                return _finish()
            token, _ = _step(
                model,
                _token_tensor([token_id], target_device),
                offset,
                kv_cache,
            )
            offset += 1
            _assert_cache_offset(kv_cache, offset)
            token_id = _token_id(token)
        return _finish()

    try:
        token_id, offset = _prefill(model, target_prompt_tokens, kv_cache, target_device)
        _assert_cache_offset(kv_cache, offset)
        if token_id in target_eos_ids:
            return _finish()
        if max_tokens == 1:
            _emit([token_id])
            return _finish()
        if proposal_length == 0:
            return _target_only(token_id, offset)

        draft_kv_cache = draft_model.create_kv_cache()
        draft_token_id, draft_offset = _prefill(
            draft_model,
            draft_prompt_tokens,
            draft_kv_cache,
            draft_device,
        )
        _assert_cache_offset(draft_kv_cache, draft_offset)
        assert offset == draft_offset
        if draft_token_id in draft_eos_ids:
            return _target_only(token_id, offset)

        def _draft_generate(last_token_id, cur_offset, limit):
            tokens: list[int] = []
            current_offset = cur_offset
            for _ in range(limit):
                token, _ = _step(
                    draft_model,
                    _token_tensor([last_token_id], draft_device),
                    current_offset,
                    draft_kv_cache,
                )
                last_token_id = _token_id(token)
                tokens.append(last_token_id)
                current_offset += 1
                if last_token_id in draft_eos_ids:
                    break
            return tokens, current_offset

        while True:
            remaining = max_tokens - emitted
            if remaining == 1:
                _emit([token_id])
                return _finish()
            draft_tokens, draft_offset = _draft_generate(
                token_id,
                draft_offset,
                min(proposal_length, remaining - 1),
            )
            _assert_cache_offset(draft_kv_cache, draft_offset)

            verification_ids = [token_id, *draft_tokens]
            new_tokens, _ = _step(
                model,
                _token_tensor(verification_ids, target_device),
                offset,
                kv_cache,
                len(verification_ids),
            )
            target_predictions = [
                int(value) for value in new_tokens.reshape(-1).tolist()
            ]
            assert len(target_predictions) == len(verification_ids)
            offset += len(verification_ids)
            _assert_cache_offset(kv_cache, offset)

            aligned_target = [token_id, *target_predictions[:-1]]
            mismatch_index = None
            terminal_index = None
            for i, (target_id, draft_id) in enumerate(
                zip(aligned_target, verification_ids, strict=True)
            ):
                if target_id != draft_id:
                    mismatch_index = i
                    break
                if target_id in target_eos_ids:
                    terminal_index = i
                    break

            if terminal_index is not None:
                _emit(aligned_target[:terminal_index])
                target_rewind = len(verification_ids) - terminal_index
                draft_rewind = len(draft_tokens) - terminal_index
                _rewind_cache(kv_cache, target_rewind)
                _rewind_cache(draft_kv_cache, draft_rewind)
                offset -= target_rewind
                draft_offset -= draft_rewind
                assert offset == draft_offset
                _assert_cache_offset(kv_cache, offset)
                _assert_cache_offset(draft_kv_cache, draft_offset)
                return _finish()

            if mismatch_index is not None:
                assert mismatch_index >= 1
                _emit(aligned_target[:mismatch_index])
                target_rewind = len(verification_ids) - mismatch_index
                draft_rewind = len(draft_tokens) - mismatch_index
                _rewind_cache(kv_cache, target_rewind)
                _rewind_cache(draft_kv_cache, draft_rewind)
                offset -= target_rewind
                draft_offset -= draft_rewind
                assert offset == draft_offset
                _assert_cache_offset(kv_cache, offset)
                _assert_cache_offset(draft_kv_cache, draft_offset)
                token_id = aligned_target[mismatch_index]
                if token_id in target_eos_ids:
                    return _finish()
                continue

            _emit(aligned_target)
            if emitted == max_tokens:
                return _finish()
            bonus_token_id = target_predictions[-1]
            if bonus_token_id in target_eos_ids:
                return _finish()
            if emitted + 1 == max_tokens:
                _emit([bonus_token_id])
                return _finish()

            _, draft_offset = _draft_generate(
                verification_ids[-1],
                draft_offset,
                1,
            )
            token_id = bonus_token_id
            assert offset == draft_offset
            _assert_cache_offset(kv_cache, offset)
            _assert_cache_offset(draft_kv_cache, draft_offset)
    finally:
        _release_kv_cache(draft_kv_cache)
        _release_kv_cache(kv_cache)
