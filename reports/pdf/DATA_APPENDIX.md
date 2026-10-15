# Data appendix — benchmax

Every table below is generated from `results/raw/runs.jsonl` by `python -m benchmax.cli report`. Figures are in `reports/figures/`. n = 3 replicates per (backend, model, quantization, workload) unless noted.

---

## Environment

| key | value |
|---|---|
| `timestamp_utc` | 2026-10-15T03:04:27Z |
| `platform` | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| `python` | 3.12.3 |
| `logical_cores` | 20 |
| `gpu` | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 610.62, 8.9 |
| `cpu` | Intel(R) Core(TM) i7-14650HX |
| `ram_gib` | 10.69 |
| `wsl_note` | running inside WSL2; visible RAM is the .wslconfig cap, which is smaller than host RAM |

## Backend versions

| key | value |
|---|---|
| `llamacpp` | {'commit': 'a0e6dc7384edb8f5d927155e1d26c10b4a310e6d', 'build': 'GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=89'} |
| `vllm` | 0.31.0 |

## Matrix (every cell and why it was or was not run)

| backend | model | quant | scheme | status | reason |
|---|---|---|---|---|---|
| llamacpp | qwen3-0.6b | 4bit | Q4_K_M (GGUF) | ready |  |
| llamacpp | qwen3-0.6b | 8bit | Q8_0 (GGUF) | ready |  |
| vllm | qwen3-0.6b | 8bit | FP8 (W8A8, e4m3) | ready |  |

## Performance rollup (all workloads combined)

Latency in ms, throughput in tok/s, memory in MiB. Averaged across all four workloads, so use the per-workload table for concurrency-specific claims.

| backend | model | quant_level | ttft_ms | ttft_sd | tpot_ms | itl_p99 | tok_s | goodput | vram_mib | n_reps |
|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp | Qwen3-0.6B | 4bit | 1,579.3 | 1,656.4 | 7.71 | 10.51 | 407.48 | 185.79 | 3831 | 6 |
| llamacpp | Qwen3-0.6B | 8bit | 1,364.2 | 1,419.1 | 7.33 | 9.36 | 433.79 | 186.00 | 3925 | 6 |
| vllm | Qwen3-0.6B | 8bit | 93.86 | 100.75 | 8.56 | 15.38 | 1,426.3 | 1,426.3 | 7896 | 6 |

## Performance by workload, backend and quantization

This is the table to cite for any concurrency-specific claim.

| workload | backend | quant_level | ttft_ms | ttft_sd | ttft_p95 | tpot_ms | tpot_sd | tpot_p95 | itl_p50 | itl_p99 | tok_s | tok_s_sd | prefill_tok_s | prefill_tok_s_std | goodput | goodput_tok_s_std | vram_mib | rss_mib | n_reps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1_single_short | llamacpp | 4bit | 69.20 | 99.38 | 69.20 | 4.02 | 0.14 | 4.02 | 3.90 | 5.66 | 223.80 | 31.51 | 87.42 | 12.31 | 223.80 | 31.51 | 3831 | 1,288.1 | 3 |
| w1_single_short | llamacpp | 8bit | 71.13 | 104.12 | 71.13 | 4.38 | 0.06 | 4.38 | 4.29 | 5.64 | 206.79 | 29.36 | 80.78 | 11.47 | 206.79 | 29.36 | 3925 | 1,321.3 | 3 |
| w1_single_short | vllm | 8bit | 26.61 | 4.60 | 26.61 | 6.60 | 0.28 | 6.60 | 6.32 | 10.87 | 147.95 | 6.04 | 57.79 | 2.36 | 147.95 | 6.04 | 7867 | 3,682.5 | 3 |
| w4_concurrent_32 | llamacpp | 4bit | 3,089.4 | 89.64 | 5,713.1 | 11.40 | 1.16 | 12.85 | 10.98 | 15.36 | 591.16 | 22.65 | 761.03 | 29.16 | 147.79 | 5.66 | 3831 | 2,296.1 | 3 |
| w4_concurrent_32 | llamacpp | 8bit | 2,657.2 | 90.68 | 5,040.3 | 10.29 | 0.25 | 11.46 | 10.02 | 13.09 | 660.79 | 19.38 | 850.67 | 24.95 | 165.20 | 4.85 | 3925 | 2,294.6 | 3 |
| w4_concurrent_32 | vllm | 8bit | 161.11 | 108.57 | 203.89 | 10.51 | 0.17 | 10.98 | 10.08 | 19.90 | 2,704.6 | 206.91 | 3,481.8 | 266.36 | 2,704.6 | 206.91 | 7896 | 3,683.2 | 3 |

## TTFT / throughput frontier

`on_frontier = True` means no other configuration beats this one on both throughput and TTFT.

| workload | backend | model | quant_level | ttft_ms | out_tok_s | tpot_ms | goodput_tok_s | on_frontier |
|---|---|---|---|---|---|---|---|---|
| w1_single_short | llamacpp | Llama-3.2-1B-Instruct | 16bit | 18.47 | 88.21 | 11.27 | 88.21 | True |
| w1_single_short | llamacpp | Llama-3.2-1B-Instruct | 4bit | 13.33 | 194.72 | 5.07 | 194.72 | True |
| w1_single_short | llamacpp | Llama-3.2-1B-Instruct | 8bit | 17.82 | 133.71 | 7.39 | 133.71 | True |
| w1_single_short | llamacpp | Llama-3.2-3B-Instruct | 4bit | 25.20 | 98.22 | 10.06 | 98.22 | False |
| w1_single_short | llamacpp | Llama-3.2-3B-Instruct | 8bit | 30.65 | 66.89 | 14.82 | 66.89 | False |
| w1_single_short | llamacpp | Qwen3-0.6B | 16bit | 15.35 | 134.69 | 7.36 | 134.69 | True |
| w1_single_short | llamacpp | Qwen3-0.6B | 4bit | 69.20 | 223.80 | 4.02 | 223.80 | False |
| w1_single_short | llamacpp | Qwen3-0.6B | 8bit | 71.13 | 206.79 | 4.38 | 206.79 | False |
| w1_single_short | llamacpp | Qwen3-1.7B | 16bit | 68.34 | 62.83 | 15.51 | 62.83 | False |
| w1_single_short | llamacpp | Qwen3-1.7B | 4bit | 20.11 | 146.11 | 6.74 | 146.11 | False |
| w1_single_short | llamacpp | Qwen3-1.7B | 8bit | 20.95 | 108.67 | 9.11 | 108.67 | False |
| w1_single_short | llamacpp | Qwen3-4B | 4bit | 29.50 | 75.00 | 13.20 | 75.00 | False |
| w1_single_short | llamacpp | Qwen3-4B | 8bit | 40.44 | 48.75 | 20.35 | 48.75 | True |
| w1_single_short | vllm | Llama-3.2-1B-Instruct | 16bit | 82.17 | 15.29 | 65.31 | 15.29 | True |
| w1_single_short | vllm | Llama-3.2-1B-Instruct | 8bit | 54.40 | 31.46 | 36.15 | 31.46 | True |
| w1_single_short | vllm | Llama-3.2-3B-Instruct | 8bit | 116.65 | 11.42 | 87.40 | 11.42 | True |
| w1_single_short | vllm | Qwen3-0.6B | 16bit | 19.77 | 130.22 | 7.59 | 130.22 | False |
| w1_single_short | vllm | Qwen3-0.6B | 8bit | 26.61 | 147.95 | 6.60 | 147.95 | False |
| w1_single_short | vllm | Qwen3-0.6B | int8 | 14.41 | 183.51 | 5.38 | 183.51 | True |
| w1_single_short | vllm | Qwen3-1.7B | 16bit | 25.07 | 63.23 | 15.74 | 63.23 | True |
| w1_single_short | vllm | Qwen3-1.7B | 8bit | 23.40 | 92.35 | 10.73 | 92.35 | False |
| w1_single_short | vllm | Qwen3-1.7B | int8 | 22.09 | 92.19 | 10.76 | 92.19 | False |
| w1_single_short | vllm | Qwen3-4B | 4bit | 28.81 | 71.55 | 13.86 | 71.55 | False |
| w1_single_short | vllm | Qwen3-4B | 8bit | 29.17 | 50.37 | 19.78 | 50.37 | True |
| w2_single_prefill | llamacpp | Llama-3.2-1B-Instruct | 16bit | 40.89 | 83.95 | 11.46 | 83.95 | True |
| w2_single_prefill | llamacpp | Llama-3.2-1B-Instruct | 4bit | 33.45 | 185.74 | 5.01 | 185.74 | False |
| w2_single_prefill | llamacpp | Llama-3.2-1B-Instruct | 8bit | 37.85 | 128.62 | 7.33 | 128.62 | True |
| w2_single_prefill | llamacpp | Llama-3.2-3B-Instruct | 4bit | 102.84 | 86.38 | 10.43 | 86.38 | False |
| w2_single_prefill | llamacpp | Llama-3.2-3B-Instruct | 8bit | 83.78 | 60.91 | 15.45 | 60.91 | False |
| w2_single_prefill | llamacpp | Qwen3-0.6B | 16bit | 39.76 | 119.04 | 7.95 | 119.04 | True |
| w2_single_prefill | llamacpp | Qwen3-0.6B | 4bit | 42.28 | 192.50 | 4.68 | 192.50 | False |
| w2_single_prefill | llamacpp | Qwen3-0.6B | 8bit | 34.42 | 185.57 | 4.97 | 185.57 | False |
| w2_single_prefill | llamacpp | Qwen3-1.7B | 16bit | 80.64 | 55.18 | 17.20 | 55.18 | True |
| w2_single_prefill | llamacpp | Qwen3-1.7B | 4bit | 56.06 | 125.30 | 7.32 | 125.30 | False |
| w2_single_prefill | llamacpp | Qwen3-1.7B | 8bit | 57.05 | 90.74 | 10.34 | 90.74 | False |
| w2_single_prefill | llamacpp | Qwen3-4B | 4bit | 132.82 | 62.62 | 14.46 | 62.62 | False |
| w2_single_prefill | llamacpp | Qwen3-4B | 8bit | 125.54 | 43.38 | 21.57 | 43.38 | True |
| w2_single_prefill | vllm | Llama-3.2-1B-Instruct | 16bit | 516.09 | 13.71 | 67.23 | 9.96 | False |
| w2_single_prefill | vllm | Llama-3.2-1B-Instruct | 8bit | 343.96 | 19.19 | 48.47 | 19.19 | True |
| w2_single_prefill | vllm | Llama-3.2-3B-Instruct | 8bit | 477.76 | 11.29 | 83.21 | 7.01 | True |
| w2_single_prefill | vllm | Qwen3-0.6B | 16bit | 39.92 | 116.59 | 8.09 | 116.59 | True |
| w2_single_prefill | vllm | Qwen3-0.6B | 8bit | 28.96 | 175.94 | 5.32 | 175.94 | False |
| w2_single_prefill | vllm | Qwen3-0.6B | int8 | 28.07 | 172.33 | 5.45 | 172.33 | True |
| w2_single_prefill | vllm | Qwen3-1.7B | 16bit | 62.53 | 59.42 | 16.13 | 59.42 | True |
| w2_single_prefill | vllm | Qwen3-1.7B | 8bit | 56.84 | 86.54 | 10.87 | 86.54 | False |
| w2_single_prefill | vllm | Qwen3-1.7B | int8 | 61.11 | 85.12 | 11.02 | 85.12 | False |
| w2_single_prefill | vllm | Qwen3-4B | 4bit | 123.28 | 62.87 | 14.42 | 62.87 | False |
| w2_single_prefill | vllm | Qwen3-4B | 8bit | 130.61 | 44.18 | 21.08 | 44.18 | False |
| w3_concurrent_8 | llamacpp | Llama-3.2-1B-Instruct | 16bit | 361.75 | 448.95 | 15.30 | 448.95 | False |
| w3_concurrent_8 | llamacpp | Llama-3.2-1B-Instruct | 4bit | 269.70 | 462.95 | 15.50 | 462.95 | False |
| w3_concurrent_8 | llamacpp | Llama-3.2-1B-Instruct | 8bit | 129.30 | 548.88 | 13.64 | 548.88 | False |
| w3_concurrent_8 | llamacpp | Llama-3.2-3B-Instruct | 4bit | 1,758.2 | 199.81 | 28.73 | 90.27 | False |
| w3_concurrent_8 | llamacpp | Llama-3.2-3B-Instruct | 8bit | 1,044.6 | 262.53 | 23.31 | 109.02 | False |
| w3_concurrent_8 | llamacpp | Qwen3-0.6B | 16bit | 185.63 | 615.11 | 11.69 | 615.11 | False |
| w3_concurrent_8 | llamacpp | Qwen3-0.6B | 4bit | 202.30 | 534.34 | 13.50 | 534.34 | False |
| w3_concurrent_8 | llamacpp | Qwen3-0.6B | 8bit | 240.80 | 504.71 | 14.10 | 504.71 | False |
| w3_concurrent_8 | llamacpp | Qwen3-1.7B | 16bit | 1,695.8 | 233.15 | 23.34 | 102.67 | False |
| w3_concurrent_8 | llamacpp | Qwen3-1.7B | 4bit | 239.91 | 388.60 | 18.94 | 388.60 | False |
| w3_concurrent_8 | llamacpp | Qwen3-1.7B | 8bit | 299.88 | 371.16 | 19.61 | 371.16 | False |
| w3_concurrent_8 | llamacpp | Qwen3-4B | 4bit | 466.59 | 187.22 | 39.38 | 187.22 | False |
| w3_concurrent_8 | llamacpp | Qwen3-4B | 8bit | 1,564.0 | 188.36 | 32.23 | 81.80 | False |
| w3_concurrent_8 | vllm | Llama-3.2-1B-Instruct | 16bit | 652.55 | 105.55 | 72.70 | 75.85 | True |
| w3_concurrent_8 | vllm | Llama-3.2-1B-Instruct | 8bit | 350.88 | 153.51 | 49.68 | 153.51 | True |
| w3_concurrent_8 | vllm | Llama-3.2-3B-Instruct | 8bit | 570.45 | 107.49 | 70.59 | 78.17 | True |
| w3_concurrent_8 | vllm | Qwen3-0.6B | 16bit | 38.87 | 1,024.5 | 7.52 | 1,024.5 | True |
| w3_concurrent_8 | vllm | Qwen3-0.6B | 8bit | 38.30 | 1,302.3 | 5.85 | 1,302.3 | False |
| w3_concurrent_8 | vllm | Qwen3-0.6B | int8 | 38.15 | 1,266.3 | 6.04 | 1,266.3 | True |
| w3_concurrent_8 | vllm | Qwen3-1.7B | 16bit | 69.38 | 470.97 | 16.53 | 470.97 | True |
| w3_concurrent_8 | vllm | Qwen3-1.7B | 8bit | 64.26 | 679.11 | 11.33 | 679.11 | True |
| w3_concurrent_8 | vllm | Qwen3-1.7B | int8 | 63.59 | 691.55 | 11.12 | 691.55 | True |
| w3_concurrent_8 | vllm | Qwen3-4B | 4bit | 129.71 | 509.56 | 14.76 | 509.56 | False |
| w3_concurrent_8 | vllm | Qwen3-4B | 8bit | 129.41 | 362.52 | 21.20 | 362.52 | True |
| w4_concurrent_32 | llamacpp | Llama-3.2-1B-Instruct | 16bit | 3,554.2 | 453.32 | 16.65 | 113.33 | False |
| w4_concurrent_32 | llamacpp | Llama-3.2-1B-Instruct | 4bit | 3,374.4 | 505.68 | 14.49 | 126.42 | False |
| w4_concurrent_32 | llamacpp | Llama-3.2-1B-Instruct | 8bit | 3,534.5 | 445.91 | 16.80 | 111.48 | False |
| w4_concurrent_32 | llamacpp | Llama-3.2-3B-Instruct | 4bit | 7,486.7 | 224.56 | 30.66 | 56.14 | True |
| w4_concurrent_32 | llamacpp | Llama-3.2-3B-Instruct | 8bit | 6,237.8 | 269.55 | 25.44 | 67.39 | True |
| w4_concurrent_32 | llamacpp | Qwen3-0.6B | 16bit | 3,674.5 | 462.63 | 14.16 | 115.66 | False |
| w4_concurrent_32 | llamacpp | Qwen3-0.6B | 4bit | 3,089.4 | 591.16 | 11.40 | 147.79 | False |
| w4_concurrent_32 | llamacpp | Qwen3-0.6B | 8bit | 2,657.2 | 660.79 | 10.29 | 165.20 | False |
| w4_concurrent_32 | llamacpp | Qwen3-1.7B | 16bit | 6,478.1 | 267.80 | 24.22 | 66.95 | True |
| w4_concurrent_32 | llamacpp | Qwen3-1.7B | 4bit | 4,152.1 | 384.77 | 19.17 | 96.19 | False |
| w4_concurrent_32 | llamacpp | Qwen3-1.7B | 8bit | 4,446.2 | 378.57 | 19.15 | 94.64 | False |
| w4_concurrent_32 | llamacpp | Qwen3-4B | 4bit | 8,680.0 | 186.38 | 40.34 | 46.60 | True |
| w4_concurrent_32 | llamacpp | Qwen3-4B | 8bit | 8,185.0 | 205.25 | 33.77 | 51.31 | True |
| w4_concurrent_32 | vllm | Llama-3.2-1B-Instruct | 16bit | 1,554.1 | 370.65 | 77.79 | 277.70 | False |
| w4_concurrent_32 | vllm | Llama-3.2-1B-Instruct | 8bit | 1,050.5 | 485.94 | 58.13 | 355.35 | True |
| w4_concurrent_32 | vllm | Llama-3.2-3B-Instruct | 8bit | 1,196.5 | 308.59 | 94.53 | 97.68 | True |
| w4_concurrent_32 | vllm | Qwen3-0.6B | 16bit | 99.27 | 2,788.7 | 10.66 | 2,788.7 | True |
| w4_concurrent_32 | vllm | Qwen3-0.6B | 8bit | 161.11 | 2,704.6 | 10.51 | 2,704.6 | True |
| w4_concurrent_32 | vllm | Qwen3-0.6B | int8 | 103.37 | 2,937.3 | 9.50 | 2,937.3 | False |
| w4_concurrent_32 | vllm | Qwen3-1.7B | 16bit | 195.43 | 1,446.3 | 20.70 | 1,446.3 | True |
| w4_concurrent_32 | vllm | Qwen3-1.7B | 8bit | 218.33 | 1,870.4 | 15.70 | 1,870.4 | False |
| w4_concurrent_32 | vllm | Qwen3-1.7B | int8 | 206.40 | 1,859.7 | 15.72 | 1,859.7 | False |
| w4_concurrent_32 | vllm | Qwen3-4B | 4bit | 434.68 | 1,260.4 | 22.72 | 1,118.5 | True |
| w4_concurrent_32 | vllm | Qwen3-4B | 8bit | 471.13 | 1,031.5 | 28.48 | 894.31 | True |

## Quantization cost — perplexity

Normalised per model against that model's own unquantized run. Cross-backend perplexity is NOT comparable (different context windows and tools).

| axis | backend | model | quant_level | quant_scheme | value | reference | ratio | workload |
|---|---|---|---|---|---|---|---|---|
| perplexity | llamacpp | Llama-3.2-1B-Instruct | 4bit | Q4_K_M (GGUF) | 10.64 | 10.42 | 1.02 |  |
| perplexity | llamacpp | Llama-3.2-1B-Instruct | 8bit | Q8_0 (GGUF) | 10.41 | 10.42 | 1.00 |  |
| perplexity | llamacpp | Llama-3.2-1B-Instruct | 16bit | F16 (GGUF) | 10.42 | 10.42 | 1.00 |  |
| perplexity | llamacpp | Qwen3-0.6B | 16bit | BF16 (GGUF) | 15.25 | 15.25 | 1.00 |  |
| perplexity | llamacpp | Qwen3-1.7B | 4bit | Q4_K_M (GGUF) | 13.37 | 13.04 | 1.03 |  |
| perplexity | llamacpp | Qwen3-1.7B | 8bit | Q8_0 (GGUF) | 13.00 | 13.04 | 1.00 |  |
| perplexity | llamacpp | Qwen3-1.7B | 16bit | BF16 (GGUF) | 13.04 | 13.04 | 1.00 |  |
| perplexity | vllm | Llama-3.2-1B-Instruct | 8bit | FP8 dynamic (W8A8) | 10.74 | 10.67 | 1.01 |  |
| perplexity | vllm | Llama-3.2-1B-Instruct | 16bit | BF16 (unquantized) | 10.67 | 10.67 | 1.00 |  |
| perplexity | vllm | Qwen3-0.6B | int8 | GPTQ-INT8 (W8A16) | 15.00 | 14.94 | 1.00 |  |
| perplexity | vllm | Qwen3-0.6B | 8bit | FP8 (W8A8, e4m3) | 15.22 | 14.94 | 1.02 |  |
| perplexity | vllm | Qwen3-0.6B | 16bit | BF16 (unquantized) | 14.94 | 14.94 | 1.00 |  |
| perplexity | vllm | Qwen3-1.7B | int8 | GPTQ-INT8 (W8A16) | 12.52 | 12.52 | 1.00 |  |
| perplexity | vllm | Qwen3-1.7B | 8bit | FP8 (W8A8, e4m3) | 12.43 | 12.52 | 0.99 |  |
| perplexity | vllm | Qwen3-1.7B | 16bit | BF16 (unquantized) | 12.52 | 12.52 | 1.00 |  |

## Quantization cost — throughput

Normalised per (backend, model) against the same configuration's 16-bit reference.

| axis | backend | model | quant_level | quant_scheme | value | reference | ratio | workload |
|---|---|---|---|---|---|---|---|---|
| throughput | llamacpp | llama32-1b | 16bit | F16 (GGUF) | 88.21 | 88.21 | 1.00 | w1_single_short |
| throughput | llamacpp | llama32-1b | 4bit | Q4_K_M (GGUF) | 194.72 | 88.21 | 2.21 | w1_single_short |
| throughput | llamacpp | llama32-1b | 8bit | Q8_0 (GGUF) | 133.71 | 88.21 | 1.52 | w1_single_short |
| throughput | llamacpp | qwen3-0.6b | 16bit | BF16 (GGUF) | 134.69 | 134.69 | 1.00 | w1_single_short |
| throughput | llamacpp | qwen3-0.6b | 4bit | Q4_K_M (GGUF) | 223.80 | 134.69 | 1.66 | w1_single_short |
| throughput | llamacpp | qwen3-0.6b | 8bit | Q8_0 (GGUF) | 206.79 | 134.69 | 1.54 | w1_single_short |
| throughput | llamacpp | qwen3-1.7b | 16bit | BF16 (GGUF) | 62.83 | 62.83 | 1.00 | w1_single_short |
| throughput | llamacpp | qwen3-1.7b | 4bit | Q4_K_M (GGUF) | 146.11 | 62.83 | 2.33 | w1_single_short |
| throughput | llamacpp | qwen3-1.7b | 8bit | Q8_0 (GGUF) | 108.67 | 62.83 | 1.73 | w1_single_short |
| throughput | vllm | llama32-1b | 16bit | BF16 (unquantized) | 15.29 | 15.29 | 1.00 | w1_single_short |
| throughput | vllm | llama32-1b | 8bit | FP8 dynamic (W8A8) | 31.46 | 15.29 | 2.06 | w1_single_short |
| throughput | vllm | qwen3-0.6b | 16bit | BF16 (unquantized) | 130.22 | 130.22 | 1.00 | w1_single_short |
| throughput | vllm | qwen3-0.6b | 8bit | FP8 (W8A8, e4m3) | 147.95 | 130.22 | 1.14 | w1_single_short |
| throughput | vllm | qwen3-0.6b | int8 | GPTQ-INT8 (W8A16) | 183.51 | 130.22 | 1.41 | w1_single_short |
| throughput | vllm | qwen3-1.7b | 16bit | BF16 (unquantized) | 63.23 | 63.23 | 1.00 | w1_single_short |
| throughput | vllm | qwen3-1.7b | 8bit | FP8 (W8A8, e4m3) | 92.35 | 63.23 | 1.46 | w1_single_short |
| throughput | vllm | qwen3-1.7b | int8 | GPTQ-INT8 (W8A16) | 92.19 | 63.23 | 1.46 | w1_single_short |
| throughput | llamacpp | llama32-1b | 16bit | F16 (GGUF) | 83.95 | 83.95 | 1.00 | w2_single_prefill |
| throughput | llamacpp | llama32-1b | 4bit | Q4_K_M (GGUF) | 185.74 | 83.95 | 2.21 | w2_single_prefill |
| throughput | llamacpp | llama32-1b | 8bit | Q8_0 (GGUF) | 128.62 | 83.95 | 1.53 | w2_single_prefill |
| throughput | llamacpp | qwen3-0.6b | 16bit | BF16 (GGUF) | 119.04 | 119.04 | 1.00 | w2_single_prefill |
| throughput | llamacpp | qwen3-0.6b | 4bit | Q4_K_M (GGUF) | 192.50 | 119.04 | 1.62 | w2_single_prefill |
| throughput | llamacpp | qwen3-0.6b | 8bit | Q8_0 (GGUF) | 185.57 | 119.04 | 1.56 | w2_single_prefill |
| throughput | llamacpp | qwen3-1.7b | 16bit | BF16 (GGUF) | 55.18 | 55.18 | 1.00 | w2_single_prefill |
| throughput | llamacpp | qwen3-1.7b | 4bit | Q4_K_M (GGUF) | 125.30 | 55.18 | 2.27 | w2_single_prefill |
| throughput | llamacpp | qwen3-1.7b | 8bit | Q8_0 (GGUF) | 90.74 | 55.18 | 1.64 | w2_single_prefill |
| throughput | vllm | llama32-1b | 16bit | BF16 (unquantized) | 13.71 | 13.71 | 1.00 | w2_single_prefill |
| throughput | vllm | llama32-1b | 8bit | FP8 dynamic (W8A8) | 19.19 | 13.71 | 1.40 | w2_single_prefill |
| throughput | vllm | qwen3-0.6b | 16bit | BF16 (unquantized) | 116.59 | 116.59 | 1.00 | w2_single_prefill |
| throughput | vllm | qwen3-0.6b | 8bit | FP8 (W8A8, e4m3) | 175.94 | 116.59 | 1.51 | w2_single_prefill |
| throughput | vllm | qwen3-0.6b | int8 | GPTQ-INT8 (W8A16) | 172.33 | 116.59 | 1.48 | w2_single_prefill |
| throughput | vllm | qwen3-1.7b | 16bit | BF16 (unquantized) | 59.42 | 59.42 | 1.00 | w2_single_prefill |
| throughput | vllm | qwen3-1.7b | 8bit | FP8 (W8A8, e4m3) | 86.54 | 59.42 | 1.46 | w2_single_prefill |
| throughput | vllm | qwen3-1.7b | int8 | GPTQ-INT8 (W8A16) | 85.12 | 59.42 | 1.43 | w2_single_prefill |
| throughput | llamacpp | llama32-1b | 16bit | F16 (GGUF) | 448.95 | 448.95 | 1.00 | w3_concurrent_8 |
| throughput | llamacpp | llama32-1b | 4bit | Q4_K_M (GGUF) | 462.95 | 448.95 | 1.03 | w3_concurrent_8 |
| throughput | llamacpp | llama32-1b | 8bit | Q8_0 (GGUF) | 548.88 | 448.95 | 1.22 | w3_concurrent_8 |
| throughput | llamacpp | qwen3-0.6b | 16bit | BF16 (GGUF) | 615.11 | 615.11 | 1.00 | w3_concurrent_8 |
| throughput | llamacpp | qwen3-0.6b | 4bit | Q4_K_M (GGUF) | 534.34 | 615.11 | 0.87 | w3_concurrent_8 |
| throughput | llamacpp | qwen3-0.6b | 8bit | Q8_0 (GGUF) | 504.71 | 615.11 | 0.82 | w3_concurrent_8 |
| throughput | llamacpp | qwen3-1.7b | 16bit | BF16 (GGUF) | 233.15 | 233.15 | 1.00 | w3_concurrent_8 |
| throughput | llamacpp | qwen3-1.7b | 4bit | Q4_K_M (GGUF) | 388.60 | 233.15 | 1.67 | w3_concurrent_8 |
| throughput | llamacpp | qwen3-1.7b | 8bit | Q8_0 (GGUF) | 371.16 | 233.15 | 1.59 | w3_concurrent_8 |
| throughput | vllm | llama32-1b | 16bit | BF16 (unquantized) | 105.55 | 105.55 | 1.00 | w3_concurrent_8 |
| throughput | vllm | llama32-1b | 8bit | FP8 dynamic (W8A8) | 153.51 | 105.55 | 1.45 | w3_concurrent_8 |
| throughput | vllm | qwen3-0.6b | 16bit | BF16 (unquantized) | 1,024.5 | 1,024.5 | 1.00 | w3_concurrent_8 |
| throughput | vllm | qwen3-0.6b | 8bit | FP8 (W8A8, e4m3) | 1,302.3 | 1,024.5 | 1.27 | w3_concurrent_8 |
| throughput | vllm | qwen3-0.6b | int8 | GPTQ-INT8 (W8A16) | 1,266.3 | 1,024.5 | 1.24 | w3_concurrent_8 |
| throughput | vllm | qwen3-1.7b | 16bit | BF16 (unquantized) | 470.97 | 470.97 | 1.00 | w3_concurrent_8 |
| throughput | vllm | qwen3-1.7b | 8bit | FP8 (W8A8, e4m3) | 679.11 | 470.97 | 1.44 | w3_concurrent_8 |
| throughput | vllm | qwen3-1.7b | int8 | GPTQ-INT8 (W8A16) | 691.55 | 470.97 | 1.47 | w3_concurrent_8 |
| throughput | llamacpp | llama32-1b | 16bit | F16 (GGUF) | 453.32 | 453.32 | 1.00 | w4_concurrent_32 |
| throughput | llamacpp | llama32-1b | 4bit | Q4_K_M (GGUF) | 505.68 | 453.32 | 1.12 | w4_concurrent_32 |
| throughput | llamacpp | llama32-1b | 8bit | Q8_0 (GGUF) | 445.91 | 453.32 | 0.98 | w4_concurrent_32 |
| throughput | llamacpp | qwen3-0.6b | 16bit | BF16 (GGUF) | 462.63 | 462.63 | 1.00 | w4_concurrent_32 |
| throughput | llamacpp | qwen3-0.6b | 4bit | Q4_K_M (GGUF) | 591.16 | 462.63 | 1.28 | w4_concurrent_32 |
| throughput | llamacpp | qwen3-0.6b | 8bit | Q8_0 (GGUF) | 660.79 | 462.63 | 1.43 | w4_concurrent_32 |
| throughput | llamacpp | qwen3-1.7b | 16bit | BF16 (GGUF) | 267.80 | 267.80 | 1.00 | w4_concurrent_32 |
| throughput | llamacpp | qwen3-1.7b | 4bit | Q4_K_M (GGUF) | 384.77 | 267.80 | 1.44 | w4_concurrent_32 |
| throughput | llamacpp | qwen3-1.7b | 8bit | Q8_0 (GGUF) | 378.57 | 267.80 | 1.41 | w4_concurrent_32 |
| throughput | vllm | llama32-1b | 16bit | BF16 (unquantized) | 370.65 | 370.65 | 1.00 | w4_concurrent_32 |
| throughput | vllm | llama32-1b | 8bit | FP8 dynamic (W8A8) | 485.94 | 370.65 | 1.31 | w4_concurrent_32 |
| throughput | vllm | qwen3-0.6b | 16bit | BF16 (unquantized) | 2,788.7 | 2,788.7 | 1.00 | w4_concurrent_32 |
| throughput | vllm | qwen3-0.6b | 8bit | FP8 (W8A8, e4m3) | 2,704.6 | 2,788.7 | 0.97 | w4_concurrent_32 |
| throughput | vllm | qwen3-0.6b | int8 | GPTQ-INT8 (W8A16) | 2,937.3 | 2,788.7 | 1.05 | w4_concurrent_32 |
| throughput | vllm | qwen3-1.7b | 16bit | BF16 (unquantized) | 1,446.3 | 1,446.3 | 1.00 | w4_concurrent_32 |
| throughput | vllm | qwen3-1.7b | 8bit | FP8 (W8A8, e4m3) | 1,870.4 | 1,446.3 | 1.29 | w4_concurrent_32 |
| throughput | vllm | qwen3-1.7b | int8 | GPTQ-INT8 (W8A16) | 1,859.7 | 1,446.3 | 1.29 | w4_concurrent_32 |

## Lifecycle — time-to-ready and peak memory

| backend | model | quant_level | quant_scheme | startup_s | gpu_peak_mib | host_rss_peak_mib |
|---|---|---|---|---|---|---|
| llamacpp | Qwen3-0.6B | 4bit | Q4_K_M (GGUF) | 2.06 | 4040 | 4,090.6 |
| llamacpp | Qwen3-0.6B | 8bit | Q8_0 (GGUF) | 7.02 | 4417 | 4,190.8 |
| llamacpp | Qwen3-0.6B | 16bit | BF16 (GGUF) | 3.01 | 5621 | 4,552.6 |
| llamacpp | Qwen3-4B | 4bit | Q4_K_M (GGUF) | 14.07 | 6724 | 8,208.9 |
| llamacpp | Qwen3-4B | 8bit | Q8_0 (GGUF) | 24.09 | 7876 | 1,602.9 |
| llamacpp | Llama-3.2-1B-Instruct | 4bit | Q4_K_M (GGUF) | 5.13 | 3977 | 1,664.7 |
| llamacpp | Llama-3.2-1B-Instruct | 8bit | Q8_0 (GGUF) | 7.02 | 4965 | 1,692.5 |
| llamacpp | Llama-3.2-1B-Instruct | 16bit | F16 (GGUF) | 16.06 | 7301 | 2,125.7 |
| llamacpp | Llama-3.2-3B-Instruct | 4bit | Q4_K_M (GGUF) | 11.07 | 7574 | 4,246.1 |
| llamacpp | Llama-3.2-3B-Instruct | 8bit | Q8_0 (GGUF) | 33.41 | 7711 | 4,422.8 |
| llamacpp | Qwen3-4B | 4bit | Q4_K_M (GGUF) | 4.08 | 7763 | 5,138.5 |
| llamacpp | Qwen3-4B | 8bit | Q8_0 (GGUF) | 26.07 | 7766 | 5,271.4 |
| vllm | Qwen3-0.6B | int8 | GPTQ-INT8 (W8A16) | 41.09 | 7883 | 3,810.8 |
| vllm | Qwen3-0.6B | 8bit | FP8 (W8A8, e4m3) | 42.05 | 7874 | 3,817.7 |
| vllm | Qwen3-0.6B | 16bit | BF16 (unquantized) | 41.06 | 7897 | 3,767.8 |
| vllm | Qwen3-1.7B | int8 | GPTQ-INT8 (W8A16) | 45.06 | 7880 | 3,843.8 |
| vllm | Qwen3-1.7B | 8bit | FP8 (W8A8, e4m3) | 73.30 | 7859 | 3,817.7 |
| vllm | Qwen3-1.7B | 16bit | BF16 (unquantized) | 79.13 | 7823 | 3,764.3 |
| vllm | Qwen3-4B | 4bit | AWQ-INT4 (W4A16) | 91.14 | 7907 | 3,932.6 |
| vllm | Qwen3-4B | 8bit | FP8 (W8A8, e4m3) | 132.19 | 7920 | 3,865.8 |
| vllm | Llama-3.2-1B-Instruct | 8bit | FP8 dynamic (W8A8) | 116.24 | 7839 | 4,044.7 |
| vllm | Llama-3.2-1B-Instruct | 16bit | BF16 (unquantized) | 99.19 | 7730 | 3,956.5 |
| vllm | Llama-3.2-3B-Instruct | 8bit | FP8 dynamic (W8A8) | 165.28 | 7595 | 4,179.5 |
| llamacpp | Qwen3-1.7B | 4bit | Q4_K_M (GGUF) | 3.01 | 5210 | 4,162.5 |
| llamacpp | Qwen3-1.7B | 8bit | Q8_0 (GGUF) | 8.02 | 6765 | 4,193.3 |
| llamacpp | Qwen3-1.7B | 16bit | BF16 (GGUF) | 27.07 | 7593 | 4,960.9 |
| llamacpp | Qwen3-0.6B | 4bit | Q4_K_M (GGUF) | 2.01 | 2993 | 623.31 |
| llamacpp | Qwen3-0.6B | 8bit | Q8_0 (GGUF) | 2.00 | 3225 | 641.17 |
| vllm | Qwen3-0.6B | 8bit | FP8 (W8A8, e4m3) | 48.08 | 7863 | 3,678.1 |
| llamacpp | Qwen3-0.6B | 4bit | Q4_K_M (GGUF) | 2.01 | 3831 | 2,391.3 |
| llamacpp | Qwen3-0.6B | 8bit | Q8_0 (GGUF) | 3.01 | 3925 | 2,387.7 |
| vllm | Qwen3-0.6B | 8bit | FP8 (W8A8, e4m3) | 53.20 | 7928 | 3,688.3 |

## Memory efficiency

`tok_s_per_mib` = output throughput per MiB of peak VRAM. `vram_mib_per_1b_params` is the footprint scaling factor.

| workload | backend | model_key | model | params_b | quant_level | quant_scheme | out_tok_s | gpu_peak_mib | host_rss_peak_mib | tok_s_per_mib | vram_mib_per_1b_params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w1_single_short | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 4bit | Q4_K_M (GGUF) | 194.72 | 2827 | 1,289.1 | 0.07 | 2,827.0 |
| w1_single_short | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 4bit | Q4_K_M (GGUF) | 223.80 | 3831 | 1,288.1 | 0.06 | 6,385.0 |
| w1_single_short | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | Q8_0 (GGUF) | 206.79 | 3925 | 1,321.3 | 0.05 | 6,541.7 |
| w1_single_short | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 4bit | Q4_K_M (GGUF) | 146.11 | 3832 | 2,978.5 | 0.04 | 2,254.1 |
| w1_single_short | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | Q8_0 (GGUF) | 133.71 | 3532 | 1,346.3 | 0.04 | 3,532.0 |
| w1_single_short | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (GGUF) | 134.69 | 3699 | 3,333.8 | 0.04 | 6,165.0 |
| w1_single_short | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | Q8_0 (GGUF) | 108.67 | 4214 | 3,030.7 | 0.03 | 2,478.8 |
| w1_single_short | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | int8 | GPTQ-INT8 (W8A16) | 183.51 | 7426 | 3,805.3 | 0.02 | 12,376.7 |
| w1_single_short | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | F16 (GGUF) | 88.21 | 4637 | 1,742.3 | 0.02 | 4,637.0 |
| w1_single_short | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | FP8 (W8A8, e4m3) | 147.95 | 7867 | 3,682.5 | 0.02 | 13,111.7 |
| w1_single_short | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (unquantized) | 130.22 | 6934 | 3,752.6 | 0.02 | 11,556.7 |
| w1_single_short | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 4bit | Q4_K_M (GGUF) | 98.22 | 5270 | 3,059.1 | 0.02 | 1,756.7 |
| w1_single_short | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | FP8 (W8A8, e4m3) | 92.35 | 6970 | 3,812.7 | 0.01 | 4,100.0 |
| w1_single_short | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | int8 | GPTQ-INT8 (W8A16) | 92.19 | 7067 | 3,838.3 | 0.01 | 4,157.1 |
| w1_single_short | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 4bit | Q4_K_M (GGUF) | 75.00 | 5916 | 3,710.3 | 0.01 | 1,479.0 |
| w1_single_short | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (GGUF) | 62.83 | 5834 | 3,661.0 | 0.01 | 3,431.8 |
| w1_single_short | vllm | qwen3-4b | Qwen3-4B | 4.00 | 4bit | AWQ-INT4 (W4A16) | 71.55 | 7150 | 3,927.0 | 0.01 | 1,787.5 |
| w1_single_short | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | Q8_0 (GGUF) | 66.89 | 6733 | 3,159.9 | 0.01 | 2,244.3 |
| w1_single_short | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (unquantized) | 63.23 | 6882 | 3,745.3 | 0.01 | 4,048.2 |
| w1_single_short | vllm | qwen3-4b | Qwen3-4B | 4.00 | 8bit | FP8 (W8A8, e4m3) | 50.37 | 7415 | 3,860.4 | 0.01 | 1,853.8 |
| w1_single_short | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 8bit | Q8_0 (GGUF) | 48.75 | 7539 | 3,810.1 | 0.01 | 1,884.8 |
| w1_single_short | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | FP8 dynamic (W8A8) | 31.46 | 7218 | 4,040.2 | 0.00 | 7,218.0 |
| w1_single_short | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | BF16 (unquantized) | 15.29 | 7331 | 3,890.9 | 0.00 | 7,331.0 |
| w1_single_short | vllm | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | FP8 dynamic (W8A8) | 11.42 | 7296 | 4,102.2 | 0.00 | 2,432.0 |
| w2_single_prefill | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 4bit | Q4_K_M (GGUF) | 185.74 | 2827 | 1,295.7 | 0.07 | 2,827.0 |
| w2_single_prefill | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 4bit | Q4_K_M (GGUF) | 192.50 | 2992 | 2,862.1 | 0.06 | 4,986.7 |
| w2_single_prefill | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | Q8_0 (GGUF) | 185.57 | 3481 | 2,906.3 | 0.05 | 5,801.7 |
| w2_single_prefill | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | Q8_0 (GGUF) | 128.62 | 3532 | 1,352.9 | 0.04 | 3,532.0 |
| w2_single_prefill | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 4bit | Q4_K_M (GGUF) | 125.30 | 3832 | 2,980.1 | 0.03 | 2,254.1 |
| w2_single_prefill | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (GGUF) | 119.04 | 3699 | 3,335.8 | 0.03 | 6,165.0 |
| w2_single_prefill | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | FP8 (W8A8, e4m3) | 175.94 | 6904 | 3,789.7 | 0.03 | 11,506.7 |
| w2_single_prefill | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | int8 | GPTQ-INT8 (W8A16) | 172.33 | 7426 | 3,805.3 | 0.02 | 12,376.7 |
| w2_single_prefill | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | Q8_0 (GGUF) | 90.74 | 4214 | 3,032.9 | 0.02 | 2,478.8 |
| w2_single_prefill | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | F16 (GGUF) | 83.95 | 4637 | 1,748.8 | 0.02 | 4,637.0 |
| w2_single_prefill | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (unquantized) | 116.59 | 6960 | 3,752.6 | 0.02 | 11,600.0 |
| w2_single_prefill | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 4bit | Q4_K_M (GGUF) | 86.38 | 5270 | 3,061.3 | 0.02 | 1,756.7 |
| w2_single_prefill | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | FP8 (W8A8, e4m3) | 86.54 | 6970 | 3,812.7 | 0.01 | 4,100.0 |
| w2_single_prefill | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | int8 | GPTQ-INT8 (W8A16) | 85.12 | 7067 | 3,838.3 | 0.01 | 4,157.1 |
| w2_single_prefill | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 4bit | Q4_K_M (GGUF) | 62.62 | 5951 | 3,712.7 | 0.01 | 1,487.8 |
| w2_single_prefill | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (GGUF) | 55.18 | 5841 | 3,663.8 | 0.01 | 3,435.9 |
| w2_single_prefill | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | Q8_0 (GGUF) | 60.91 | 6733 | 3,162.9 | 0.01 | 2,244.3 |
| w2_single_prefill | vllm | qwen3-4b | Qwen3-4B | 4.00 | 4bit | AWQ-INT4 (W4A16) | 62.87 | 7150 | 3,927.0 | 0.01 | 1,787.5 |
| w2_single_prefill | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (unquantized) | 59.42 | 6882 | 3,745.3 | 0.01 | 4,048.2 |
| w2_single_prefill | vllm | qwen3-4b | Qwen3-4B | 4.00 | 8bit | FP8 (W8A8, e4m3) | 44.18 | 7415 | 3,860.4 | 0.01 | 1,853.8 |
| w2_single_prefill | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 8bit | Q8_0 (GGUF) | 43.38 | 7539 | 3,812.4 | 0.01 | 1,884.8 |
| w2_single_prefill | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | FP8 dynamic (W8A8) | 19.19 | 7218 | 4,040.3 | 0.00 | 7,218.0 |
| w2_single_prefill | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | BF16 (unquantized) | 13.71 | 7331 | 3,890.9 | 0.00 | 7,331.0 |
| w2_single_prefill | vllm | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | FP8 dynamic (W8A8) | 11.29 | 7296 | 4,102.2 | 0.00 | 2,432.0 |
| w3_concurrent_8 | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | FP8 (W8A8, e4m3) | 1,302.3 | 7076 | 3,789.7 | 0.18 | 11,793.3 |
| w3_concurrent_8 | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 4bit | Q4_K_M (GGUF) | 534.34 | 2992 | 3,328.9 | 0.18 | 4,986.7 |
| w3_concurrent_8 | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | int8 | GPTQ-INT8 (W8A16) | 1,266.3 | 7426 | 3,805.4 | 0.17 | 12,376.7 |
| w3_concurrent_8 | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (GGUF) | 615.11 | 3699 | 3,810.5 | 0.17 | 6,165.0 |
| w3_concurrent_8 | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | Q8_0 (GGUF) | 548.88 | 3532 | 1,484.3 | 0.16 | 3,532.0 |
| w3_concurrent_8 | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 4bit | Q4_K_M (GGUF) | 462.95 | 3019 | 1,434.9 | 0.15 | 3,019.0 |
| w3_concurrent_8 | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (unquantized) | 1,024.5 | 6981 | 3,752.6 | 0.15 | 11,635.0 |
| w3_concurrent_8 | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | Q8_0 (GGUF) | 504.71 | 3481 | 3,418.1 | 0.14 | 5,801.7 |
| w3_concurrent_8 | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 4bit | Q4_K_M (GGUF) | 388.60 | 3832 | 3,454.0 | 0.10 | 2,254.1 |
| w3_concurrent_8 | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | int8 | GPTQ-INT8 (W8A16) | 691.55 | 7067 | 3,838.3 | 0.10 | 4,157.1 |
| w3_concurrent_8 | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | FP8 (W8A8, e4m3) | 679.11 | 6970 | 3,812.7 | 0.10 | 4,100.0 |
| w3_concurrent_8 | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | F16 (GGUF) | 448.95 | 4637 | 1,880.1 | 0.10 | 4,637.0 |
| w3_concurrent_8 | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | Q8_0 (GGUF) | 371.16 | 4494 | 3,509.2 | 0.08 | 2,643.5 |
| w3_concurrent_8 | vllm | qwen3-4b | Qwen3-4B | 4.00 | 4bit | AWQ-INT4 (W4A16) | 509.56 | 7150 | 3,927.0 | 0.07 | 1,787.5 |
| w3_concurrent_8 | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (unquantized) | 470.97 | 6882 | 3,745.3 | 0.07 | 4,048.2 |
| w3_concurrent_8 | vllm | qwen3-4b | Qwen3-4B | 4.00 | 8bit | FP8 (W8A8, e4m3) | 362.52 | 7415 | 3,860.4 | 0.05 | 1,853.8 |
| w3_concurrent_8 | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (GGUF) | 233.15 | 5916 | 4,131.0 | 0.04 | 3,480.0 |
| w3_concurrent_8 | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | Q8_0 (GGUF) | 262.53 | 6733 | 3,657.3 | 0.04 | 2,244.3 |
| w3_concurrent_8 | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 4bit | Q4_K_M (GGUF) | 199.81 | 5270 | 3,534.7 | 0.04 | 1,756.7 |
| w3_concurrent_8 | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 4bit | Q4_K_M (GGUF) | 187.22 | 5951 | 4,313.8 | 0.03 | 1,487.8 |
| w3_concurrent_8 | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 8bit | Q8_0 (GGUF) | 188.36 | 7545 | 4,447.4 | 0.02 | 1,886.2 |
| w3_concurrent_8 | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | FP8 dynamic (W8A8) | 153.51 | 7218 | 4,040.3 | 0.02 | 7,218.0 |
| w3_concurrent_8 | vllm | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | FP8 dynamic (W8A8) | 107.49 | 7389 | 4,102.2 | 0.01 | 2,463.0 |
| w3_concurrent_8 | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | BF16 (unquantized) | 105.55 | 7331 | 3,890.9 | 0.01 | 7,331.0 |
| w4_concurrent_32 | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (unquantized) | 2,788.7 | 6981 | 3,753.2 | 0.40 | 11,635.0 |
| w4_concurrent_32 | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | int8 | GPTQ-INT8 (W8A16) | 2,937.3 | 7426 | 3,805.9 | 0.40 | 12,376.7 |
| w4_concurrent_32 | vllm | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | FP8 (W8A8, e4m3) | 2,704.6 | 7896 | 3,683.2 | 0.34 | 13,160.0 |
| w4_concurrent_32 | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | FP8 (W8A8, e4m3) | 1,870.4 | 6970 | 3,813.0 | 0.27 | 4,100.0 |
| w4_concurrent_32 | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | int8 | GPTQ-INT8 (W8A16) | 1,859.7 | 7067 | 3,838.6 | 0.26 | 4,157.1 |
| w4_concurrent_32 | vllm | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (unquantized) | 1,446.3 | 6882 | 3,745.7 | 0.21 | 4,048.2 |
| w4_concurrent_32 | vllm | qwen3-4b | Qwen3-4B | 4.00 | 4bit | AWQ-INT4 (W4A16) | 1,260.4 | 7150 | 3,927.6 | 0.18 | 1,787.5 |
| w4_concurrent_32 | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 8bit | Q8_0 (GGUF) | 660.79 | 3925 | 2,294.6 | 0.17 | 6,541.7 |
| w4_concurrent_32 | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 4bit | Q4_K_M (GGUF) | 505.68 | 3019 | 1,664.7 | 0.17 | 3,019.0 |
| w4_concurrent_32 | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 4bit | Q4_K_M (GGUF) | 591.16 | 3831 | 2,296.1 | 0.15 | 6,385.0 |
| w4_concurrent_32 | vllm | qwen3-4b | Qwen3-4B | 4.00 | 8bit | FP8 (W8A8, e4m3) | 1,031.5 | 7415 | 3,860.8 | 0.14 | 1,853.8 |
| w4_concurrent_32 | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | Q8_0 (GGUF) | 445.91 | 3532 | 1,692.5 | 0.13 | 3,532.0 |
| w4_concurrent_32 | llamacpp | qwen3-0.6b | Qwen3-0.6B | 0.60 | 16bit | BF16 (GGUF) | 462.63 | 3699 | 4,552.6 | 0.13 | 6,165.0 |
| w4_concurrent_32 | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 4bit | Q4_K_M (GGUF) | 384.77 | 3832 | 4,162.5 | 0.10 | 2,254.1 |
| w4_concurrent_32 | llamacpp | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | F16 (GGUF) | 453.32 | 4637 | 2,125.7 | 0.10 | 4,637.0 |
| w4_concurrent_32 | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 8bit | Q8_0 (GGUF) | 378.57 | 4507 | 4,193.3 | 0.08 | 2,651.2 |
| w4_concurrent_32 | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 8bit | FP8 dynamic (W8A8) | 485.94 | 7218 | 4,040.9 | 0.07 | 7,218.0 |
| w4_concurrent_32 | vllm | llama32-1b | Llama-3.2-1B-Instruct | 1.00 | 16bit | BF16 (unquantized) | 370.65 | 7331 | 3,891.6 | 0.05 | 7,331.0 |
| w4_concurrent_32 | llamacpp | qwen3-1.7b | Qwen3-1.7B | 1.70 | 16bit | BF16 (GGUF) | 267.80 | 5980 | 4,960.9 | 0.04 | 3,517.6 |
| w4_concurrent_32 | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 4bit | Q4_K_M (GGUF) | 224.56 | 5270 | 4,246.1 | 0.04 | 1,756.7 |
| w4_concurrent_32 | vllm | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | FP8 dynamic (W8A8) | 308.59 | 7389 | 4,102.7 | 0.04 | 2,463.0 |
| w4_concurrent_32 | llamacpp | llama32-3b | Llama-3.2-3B-Instruct | 3.00 | 8bit | Q8_0 (GGUF) | 269.55 | 6733 | 4,422.8 | 0.04 | 2,244.3 |
| w4_concurrent_32 | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 4bit | Q4_K_M (GGUF) | 186.38 | 5951 | 5,138.5 | 0.03 | 1,487.8 |
| w4_concurrent_32 | llamacpp | qwen3-4b | Qwen3-4B | 4.00 | 8bit | Q8_0 (GGUF) | 205.25 | 7568 | 5,271.4 | 0.03 | 1,892.0 |

## Quality

`ppl_method` distinguishes llama.cpp's offline perplexity tool from vLLM's server-side echo path — they are different measurements and must not be compared directly.

| backend | model | quant_level | quant_scheme | ppl | ppl_stderr | ppl_context_tokens | ppl_method | mc_acc | mc_n | mc_wilson_lo | mc_wilson_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp | Qwen3-0.6B | 4bit | Q4_K_M (GGUF) |  |  |  | server echo logprobs |  |  |  |  |
| llamacpp | Qwen3-0.6B | 8bit | Q8_0 (GGUF) |  |  |  | server echo logprobs |  |  |  |  |
| vllm | Qwen3-0.6B | 8bit | FP8 (W8A8, e4m3) | 15.22 |  |  | server echo logprobs | 0.20 | 61.00 | 0.12 | 0.31 |

## Capability gaps

Recorded rather than hidden: these are properties of the backends, not failures of the harness.

| category | backend | model | quant_level | detail |
|---|---|---|---|---|
| mc_unavailable | llamacpp | qwen3-0.6b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-0.6b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-0.6b | 16bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-4b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-4b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | llama32-1b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | llama32-1b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | llama32-1b | 16bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | llama32-3b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | llama32-3b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-4b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-4b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| startup_failure | vllm | qwen3-0.6b | int8 |  |
| startup_failure | vllm | qwen3-0.6b | 8bit |  |
| startup_failure | vllm | qwen3-0.6b | 16bit |  |
| startup_failure | vllm | qwen3-1.7b | int8 |  |
| startup_failure | vllm | qwen3-1.7b | 8bit |  |
| startup_failure | vllm | qwen3-1.7b | 16bit |  |
| startup_failure | vllm | qwen3-4b | 4bit |  |
| startup_failure | vllm | qwen3-4b | 8bit |  |
| startup_failure | vllm | llama32-1b | 8bit |  |
| startup_failure | vllm | llama32-1b | 16bit |  |
| startup_failure | vllm | llama32-3b | 8bit |  |
| startup_failure | vllm_gguf | qwen3-4b | 4bit |  |
| startup_failure | vllm_gguf | llama32-1b | 4bit |  |
| startup_failure | vllm_gguf | llama32-3b | 4bit |  |
| mc_unavailable | llamacpp | qwen3-1.7b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-1.7b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| mc_unavailable | llamacpp | qwen3-1.7b | 16bit | top-logprobs not served by llama.cpp /v1/completions |
| perplexity_unavailable | llamacpp | qwen3-0.6b | 4bit | error |
| mc_unavailable | llamacpp | qwen3-0.6b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| perplexity_unavailable | llamacpp | qwen3-0.6b | 8bit | error |
| mc_unavailable | llamacpp | qwen3-0.6b | 8bit | top-logprobs not served by llama.cpp /v1/completions |
| perplexity_unavailable | llamacpp | qwen3-0.6b | 4bit | error |
| mc_unavailable | llamacpp | qwen3-0.6b | 4bit | top-logprobs not served by llama.cpp /v1/completions |
| perplexity_unavailable | llamacpp | qwen3-0.6b | 8bit | error |
| mc_unavailable | llamacpp | qwen3-0.6b | 8bit | top-logprobs not served by llama.cpp /v1/completions |

## Why there is no same-weights comparison

This was the intended anchor of the whole benchmark and it could not be produced.

**Reason.** vLLM 0.31 does not support GGUF. llama.cpp consumes GGUF (Q4_K_M/Q8_0/BF16); vLLM consumes safetensors with AWQ, FP8, GPTQ or compressed-tensors. The two quantization surfaces are disjoint, so 'same weights, different backend' cannot be measured by construction. Any llama.cpp-vs-vLLM throughput delta therefore confounds backend with quantization scheme and model artifact.

Verified by:

- no 'gguf' key in vllm QUANTIZATION_METHODS
- ModuleNotFoundError: vllm.model_executor.layers.quantization.gguf
- ImportError: cannot import GGUFConfig from vllm.transformers_utils.config
- serving a local .gguf path -> OSError 'not a valid JSON file'
- serving 'repo/file.gguf' -> OSError malformed repo id
