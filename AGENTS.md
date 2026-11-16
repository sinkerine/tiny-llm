# AGENTS.md

## Scope

- This file applies to the entire repository.
- Use this as the default test-running policy for coding agents.

## Objective

- Run and verify tests in a way that matches the book workflow (`book/src/*.md`).
- Prefer `pdm` entrypoints defined in `pyproject.toml`.

## Environment Requirements

- This branch targets Linux/CUDA for Weeks 1–2; see `CUDA.md`.
- Keep all student exercises unfinished. Port existing reference solutions only.
- Enter `nix develop` (or use direnv), then install dependencies:

```bash
uv sync --locked
pdm run check-installation
```

- Weeks 3–4 retain MLX/Metal and are outside the CUDA test scope.
- Week 2 uses PyTorch and native C++/CUDA extensions. Day 1 needs no native build.
- Before native Day 2–5 tests, run `nix develop --command pdm run build-ext-ref`
  for reference checks and `pdm run build-ext` for learner checks. The latter
  builds fail-closed learner stubs until the student implements them.
- CUDA extension builds use one compiler worker by default and store outputs at
  `${XDG_CACHE_HOME:-$HOME/.cache}/dev-scratch/tiny-llm/cuda-week2/{reference,student}/`.
  Rebuild and cleanup use these same directories; never delete them during an active build.
- Active Week 2 measurements use `pdm run bench-week2-cuda` (request time and
  per-request peak allocated CUDA bytes) and `pdm run profile-week2-kernels`
  (public operator replay). Reference `selected` extra scheduling has incomplete
  public replay coverage; use actual model traces instead of attributing it to
  teaching operators. Older MLX capture entrypoints remain historical.
- Include `tests_refsol/test_week_2_reference_runtime.py` in reference runtime
  checks after changing selected scheduling. These are reference-only controls,
  not additional learner exercises. CUDA launches eagerly; do not copy MLX
  graph-rewrite or submit-at-layer-18 flush mechanisms into the CUDA runtime.

- Optional baseline check from the setup chapter (reference solution, Week 1):

```bash
python -m pytest tests_refsol/test_week_1_day_*.py -q
```

## Agent Test Workflow

1. Start with the smallest relevant scope (`--week` + `--day`).
2. Use pytest filters via `-- -k ...` to isolate failing tasks.
3. Run broader suites only after targeted tests pass.
4. If extension code changed, rebuild extensions before testing.

## Canonical Commands

Run all tests:

```bash
pdm run test
```

Run a specific chapter/day:

```bash
pdm run test --week <WEEK> --day <DAY>
```

Run with pytest filters:

```bash
pdm run test --week 1 --day 3 -- -k task_2
pdm run test --week 2 --day 3 -- -k task_3
pdm run test --week 2 --day 3 -- -k gpu
```

Run reference-solution tests:

```bash
pdm run test-refsol
pdm run test-refsol --week 2 --day 3 -- -k task_3
```

## Extension Rebuild Rule

Rebuild before tests if these changed:

- `src/cuda/*` (active Week 2 CUDA kernels/bindings)
- `src/extensions/src/*` (historical Metal only)

Commands:

```bash
pdm run build-ext
```

## Guardrails

- Use `--` before pytest args (`-k`, `-q`, `--collect-only`, etc.).
- `pdm run test --week X --day Y` auto-copies `tests_refsol/test_week_X_day_Y.py` into `tests/`.
- Model-dependent tests (0.5B/1.5B/7B) skip when models are not downloaded locally.
## Model Cache

- Always download Hugging Face models into `~/.cache/tiny-llm`.
- Use `hf download Qwen/Qwen3-0.6B --cache-dir ~/.cache/tiny-llm`
  (and the same cache directory for other models).
- The Nix shell sets `HF_HUB_CACHE` to this directory. Model-backed tests must
  resolve checkpoints from this cache, including when run outside the shell.

## GitHub CLI

- The `gh` CLI is installed, authenticated, and expected to work in this
  repository.
- If a `gh` command fails because of sandbox or network restrictions, retry it
  with the required authorization. Do not infer that GitHub or `gh` is
  unavailable from a sandboxed failure.
