# Flash-MoE MiniMax M2.7 Branch

Made with Codex/GPT-5.4

This branch packages the MiniMax M2.7 work that was built on top of the original
`flash-moe` repository.

The upstream project is centered on Qwen3.5-397B. This branch is deliberately
separate and carries:

- MiniMax M2.7 checkpoint conversion scripts
- a MiniMax-specific `metal_infer/infer.m` runtime
- a reproducible OpenAI-style `/v1/chat/completions` server path
- notes and tooling needed to rebuild the q4 runtime artifacts locally

## Status

What works:

- MiniMax M2.7 dense q4 conversion
- MiniMax M2.7 routed expert q4 repack
- tokenizer/vocabulary export
- local Metal inference
- OpenAI-style serve mode with tool calling

What does not currently meet a “daily-driver” bar:

- very long cold-prefill latency
- practical high-throughput serving for huge contexts

In other words: this branch is useful as a working MiniMax implementation and a
record of what was learned, not as a finished production system.

## Quick Performance Summary

These measurements were taken on:

- MacBook Pro, 14-inch (November 2024)
- Apple M4 Pro
- 48 GB unified memory
- 512 GB internal SSD
- MiniMax M2.7 weights stored on the internal SSD

Representative numbers from the reliable OpenAI-style server path:

| Scenario | Prompt tokens | Time | Approx. throughput |
| --- | ---: | ---: | ---: |
| Base system snapshot | 39 | 18.0 s | 2.2 tok/s |
| First tool-enabled base prompt | 292 | 105.7 s | 2.8 tok/s |
| Tool-response continuation prompt | 270 | 105.2 s | 2.6 tok/s |
| Decode / generation | n/a | n/a | about 2.5 tok/s |

What this means in practice:

- Short prompts and local experimentation work.
- Tool calling works.
- Response fidelity is imperfect: the model may miscopy filenames, duplicate
  entries from tool output, or introduce small textual glitches such as stray
  characters in otherwise coherent sentences. This is likely a mix of q4
  quantization, reduced routing (`--k 4`), and lossy tool-response formatting.
- Very large cold-prefills do not currently reach a practical daily-driver bar.
- This branch is best treated as a reproducible MiniMax M2.7 port and research
  record, not a high-throughput long-context serving solution.

## Repository Layout

The files you need for the MiniMax path are:

- [metal_infer/infer.m](metal_infer/infer.m)
- [metal_infer/shaders.metal](metal_infer/shaders.metal)
- [metal_infer/Makefile](metal_infer/Makefile)
- [metal_infer/export_tokenizer_artifacts.py](metal_infer/export_tokenizer_artifacts.py)
- [metal_infer/extract_weights_minimax.py](metal_infer/extract_weights_minimax.py)
- [metal_infer/quantize_dense_minimax.py](metal_infer/quantize_dense_minimax.py)
- [repack_experts_minimax.py](repack_experts_minimax.py)
- [repack_experts_minimax_quant.py](repack_experts_minimax_quant.py)
- [quant_utils.py](quant_utils.py)
- [safetensors_utils.py](safetensors_utils.py)
- [docs/minimax-conversion.md](docs/minimax-conversion.md)

Generated artifacts such as `model_weights.bin`, `layer_XX.bin`, `vocab.bin`,
and `tokenizer.bin` are intentionally **not** kept in the repo.

## Prerequisites

- Apple Silicon Mac with Metal support
- Python 3.10+ for checkpoint conversion
- `numpy` for dense/expert quantization
- a local Hugging Face MiniMax M2.7 checkpoint directory containing:
  - `config.json`
  - `tokenizer.json`
  - `model.safetensors.index.json`
  - the referenced `*.safetensors` shard files

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy
```

## Rebuild The Runtime Artifacts

Assume the raw MiniMax checkpoint lives at:

```bash
MODEL=/path/to/MiniMax-M2.7
```

### 1. Export tokenizer artifacts

```bash
cd /path/to/flash-moe

python3 metal_infer/export_tokenizer_artifacts.py "$MODEL"
```

This writes:

- `$MODEL/tokenizer.bin`
- `$MODEL/vocab.bin`

### 2. Quantize the dense non-expert weights to q4

```bash
python3 metal_infer/quantize_dense_minimax.py --model "$MODEL"
```

This writes:

- `$MODEL/flash_moe_dense_q4/model_weights.bin`
- `$MODEL/flash_moe_dense_q4/model_weights.json`

### 3. Quantize and repack the routed experts to q4

```bash
python3 repack_experts_minimax_quant.py --model "$MODEL" --bits 4
```

This writes:

- `$MODEL/packed_experts_minimax_q4/layout.json`
- `$MODEL/packed_experts_minimax_q4/layer_00.bin`
- `...`
- `$MODEL/packed_experts_minimax_q4/layer_61.bin`

### 4. Optional native extraction helpers

These are useful for inspection and debugging, but are not required by the q4
runtime:

```bash
python3 metal_infer/extract_weights_minimax.py --model "$MODEL" --output "$MODEL/flash_moe_dense_native"
python3 repack_experts_minimax.py --model "$MODEL"
```

## Build `infer`

```bash
cd /path/to/flash-moe/metal_infer
make infer
```

## Run The Stable Server

Recommended stable command:

```bash
./infer --model "$MODEL" --serve 8000 --k 4
```

This serves:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

If you want raw request capture for debugging:

```bash
./infer \
  --model "$MODEL" \
  --serve 8000 \
  --k 4 \
  --dump-request-dir /tmp/minimax_request_dumps
```

If GPU-assisted full-attention prefill ever behaves badly on a particular
machine, force the conservative fallback:

```bash
./infer --model "$MODEL" --serve 8000 --k 4 --cpu-full-attn
```

## Experimental Flags

These are kept in the branch because they were part of the investigation, but
they are **not** the recommended default:

- `--prefill-microbatch N`
- `--prefill-microbatch-exec`
- `--prefill-parity-request PATH`
- `--prefill-microbatch-parity-request PATH`
- `--serve-replay-request PATH`

They are useful for replaying dumped requests and comparing serial vs
experimental prefill paths, but the default server path above is the reliable
one to publish.

## Expected Model Directory Layout

After running the conversion scripts, `"$MODEL"` should contain at least:

```text
MiniMax-M2.7/
  config.json
  tokenizer.json
  tokenizer.bin
  vocab.bin
  model.safetensors.index.json
  *.safetensors
  flash_moe_dense_q4/
    model_weights.bin
    model_weights.json
  packed_experts_minimax_q4/
    layout.json
    layer_00.bin
    ...
    layer_61.bin
```

## Known Limitations

- Cold prefill is still too slow for very large contexts.
- This branch is valuable as a reproducible MiniMax port, not as proof of a
  practical high-throughput `60k+` context serving system.

## More Detail

For the lower-level conversion notes and script behavior, see
[docs/minimax-conversion.md](docs/minimax-conversion.md).
