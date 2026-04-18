# MiniMax M2.7 Conversion And Runtime Notes

This document records the MiniMax-specific scripts and the artifact flow used by
the MiniMax M2.7 branch of `flash-moe`.

The short version is:

1. export `tokenizer.bin` and `vocab.bin`
2. quantize dense non-expert weights into `flash_moe_dense_q4/`
3. quantize routed experts into `packed_experts_minimax_q4/`
4. build `metal_infer/infer`
5. run the local server

## Input Checkpoint

The scripts assume a local Hugging Face-style snapshot directory with:

- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json`
- the safetensors shards referenced by the index

Examples in this document use:

```bash
MODEL=/path/to/MiniMax-M2.7
```

## Required Python Tools

- Python 3.10+
- `numpy`

Example:

```bash
cd /path/to/flash-moe
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy
```

## 1. Tokenizer Export

Script:

- [metal_infer/export_tokenizer_artifacts.py](../metal_infer/export_tokenizer_artifacts.py)

Command:

```bash
python3 metal_infer/export_tokenizer_artifacts.py "$MODEL"
```

Outputs:

- `$MODEL/tokenizer.bin`
- `$MODEL/vocab.bin`

These are the binary tokenizer artifacts consumed directly by `infer.m`.

## 2. Dense Non-Expert Conversion

There are two dense-side scripts.

### Recommended runtime artifact: q4 dense bundle

Script:

- [metal_infer/quantize_dense_minimax.py](../metal_infer/quantize_dense_minimax.py)

Command:

```bash
python3 metal_infer/quantize_dense_minimax.py --model "$MODEL"
```

Default output:

- `$MODEL/flash_moe_dense_q4/model_weights.bin`
- `$MODEL/flash_moe_dense_q4/model_weights.json`

This is the format the MiniMax runtime expects by default.

### Optional debugging artifact: native dense extraction

Script:

- [metal_infer/extract_weights_minimax.py](../metal_infer/extract_weights_minimax.py)

Command:

```bash
python3 metal_infer/extract_weights_minimax.py \
  --model "$MODEL" \
  --output "$MODEL/flash_moe_dense_native"
```

This copies non-expert tensors byte-for-byte from the safetensors shards into a
single mmap-friendly binary/manifest pair. It is useful for inspection and
parity work, but not required by the q4 runtime.

## 3. Expert Repacking

There are also two expert-side scripts.

### Recommended runtime artifact: q4 expert pack

Script:

- [repack_experts_minimax_quant.py](../repack_experts_minimax_quant.py)

Command:

```bash
python3 repack_experts_minimax_quant.py --model "$MODEL" --bits 4
```

Default output:

- `$MODEL/packed_experts_minimax_q4/layout.json`
- `$MODEL/packed_experts_minimax_q4/layer_00.bin`
- `...`
- `$MODEL/packed_experts_minimax_q4/layer_61.bin`

### Optional debugging artifact: native expert pack

Script:

- [repack_experts_minimax.py](../repack_experts_minimax.py)

Command:

```bash
python3 repack_experts_minimax.py --model "$MODEL"
```

Default output:

- `$MODEL/packed_experts_minimax_native/layout.json`
- `$MODEL/packed_experts_minimax_native/layer_00.bin`
- `...`

This native format is useful for checkpoint inspection and alternate conversion
workflows.

## 4. Build The Runtime

Files:

- [metal_infer/infer.m](../metal_infer/infer.m)
- [metal_infer/shaders.metal](../metal_infer/shaders.metal)
- [metal_infer/Makefile](../metal_infer/Makefile)

Command:

```bash
cd /path/to/flash-moe/metal_infer
make infer
```

## 5. Start The Local Server

Stable default:

```bash
./infer --model "$MODEL" --serve 8000 --k 4
```

Useful variants:

```bash
# Dump incoming OpenAI-style request bodies
./infer --model "$MODEL" --serve 8000 --k 4 --dump-request-dir /tmp/minimax_request_dumps

# Force CPU full attention if GPU-assisted prefill misbehaves on a machine
./infer --model "$MODEL" --serve 8000 --k 4 --cpu-full-attn
```

The server exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Expected Final Model Layout

After conversion, the model directory should look roughly like this:

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

## Related Files

- [README.md](../README.md)
- [quant_utils.py](../quant_utils.py)
- [safetensors_utils.py](../safetensors_utils.py)
