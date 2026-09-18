import { QUANT_OPTIONS } from '../data/presets.js';

export function calculateVramAndRecommendations(config) {
  const {
    gpuVramGb, // per GPU
    gpuCount,  // TP
    gpuMemoryUtilization, // e.g. 0.90
    paramsB,
    layers,
    hiddenSize,
    attnHeads,
    kvHeads,
    maxModelLen,
    quantizationId,
    kvCacheDtype, // 'auto' or 'fp8'
    enforceEager,
    enablePrefixCaching,
  } = config;

  const quantInfo = QUANT_OPTIONS.find(q => q.id === quantizationId) || QUANT_OPTIONS[0];
  const bytesPerParam = quantInfo.bytesPerParam;

  // 1. Model Weights Memory
  // Model weights are split across TP GPUs
  const directFileSize = Number(config.customFileSizeGb || config.safetensorsSizeGb || 0);
  let totalModelWeightGb;
  let isDirectFileMode = false;

  if (config.useDirectFileSize && directFileSize > 0) {
    totalModelWeightGb = directFileSize;
    isDirectFileMode = true;
  } else {
    totalModelWeightGb = (paramsB * 1e9 * bytesPerParam) / (1024 ** 3);
  }

  const weightPerGpuGb = totalModelWeightGb / gpuCount;
  // PyTorch / weight loading buffer overhead (~5%)
  const weightWithOverheadPerGpuGb = weightPerGpuGb * 1.05;

  // 2. Base Runtime Overhead
  // CUDA Context & PyTorch overhead
  const cudaRuntimeGb = 1.0;
  // CUDA Graph capture overhead (if eager is false, typically 1.0 - 2.0GB)
  const cudaGraphGb = enforceEager ? 0 : 1.2;
  const baseOverheadPerGpuGb = cudaRuntimeGb + cudaGraphGb;

  // 3. VRAM Budgets
  const totalVramPerGpu = gpuVramGb;
  const usableVramPerGpu = totalVramPerGpu * gpuMemoryUtilization;
  const vllmRemainingForKvGb = usableVramPerGpu - weightWithOverheadPerGpuGb - baseOverheadPerGpuGb;
  const safetyMarginGb = totalVramPerGpu * (1 - gpuMemoryUtilization);

  // 4. KV Cache Math (per token)
  const kvDtypeBytes = kvCacheDtype === 'fp8' ? 1.0 : 2.0;
  let kvBytesPerTokenPerGpu;

  // In Hybrid Linear Attention (e.g. GLM-5 Next / GLM-5.3-Flash),
  // Linear Attention layers maintain constant O(1) recurrent states without growing KV Cache.
  // ONLY full attention layers have O(N) sequence KV Cache!
  const effectiveKvLayers = (config.isHybridLinear && config.fullAttnLayers)
    ? config.fullAttnLayers
    : layers;

  if (config.isMla) {
    // Multi-Head Latent Attention (MLA)
    // Caches compressed latent vector (kv_lora_rank = 512) + decoupled key (qk_rope_head_dim)
    const mlaDim = (config.kvLoraRank || 512) + (config.qkRopeHeadDim || 0);
    kvBytesPerTokenPerGpu = effectiveKvLayers * mlaDim * kvDtypeBytes;
  } else {
    // Standard MHA / GQA (Grouped Query Attention)
    const headDim = config.headDim || (hiddenSize / attnHeads);
    const kvHeadsPerGpu = Math.max(1, kvHeads / gpuCount);
    // 2 (Key + Value) * layers * kvHeadsPerGpu * headDim * kvDtypeBytes
    kvBytesPerTokenPerGpu = 2 * effectiveKvLayers * kvHeadsPerGpu * headDim * kvDtypeBytes;
  }
  const kvKbPerTokenPerGpu = kvBytesPerTokenPerGpu / 1024;

  // Max KV tokens that can fit in remaining VRAM on each GPU
  const remainingKvBytes = Math.max(0, vllmRemainingForKvGb * (1024 ** 3));
  const maxKvTokens = Math.floor(remainingKvBytes / kvBytesPerTokenPerGpu);

  // Concurrency at maxModelLen
  const maxConcurrencyAtMaxLen = maxModelLen > 0 ? (maxKvTokens / maxModelLen) : 0;

  // 5. Status & Warnings
  const warnings = [];
  const recommendations = [];
  const errors = [];

  let status = 'optimal'; // 'optimal', 'tight', 'oom'

  if (vllmRemainingForKvGb < 0) {
    status = 'oom';
    errors.push(`【OOM エラー】モデル重み（${weightWithOverheadPerGpuGb.toFixed(1)} GB）と基盤オーバーヘッド（${baseOverheadPerGpuGb.toFixed(1)} GB）が、GPUの割り当て可能VRAM（${usableVramPerGpu.toFixed(1)} GB）を超過しています。`);
    recommendations.push(`GPU枚数（TP: Tensor Parallel）を増やすか、AWQ / GPTQ / FP8 などの量子化モデルを使用してください。`);
  } else if (vllmRemainingForKvGb < 1.5 || maxConcurrencyAtMaxLen < 1.0) {
    status = 'tight';
    warnings.push(`【VRAM逼迫】KVキャッシュ領域が ${vllmRemainingForKvGb.toFixed(2)} GB しか残っていません。max-model-len=${maxModelLen} での同時処理数が 1 未満です。`);
    if (!enforceEager) {
      recommendations.push(`--enforce-eager を有効にすると、CUDA Graphメモリ（約1.2GB）を節約してKVキャッシュに回せます。`);
    }
    if (kvCacheDtype !== 'fp8') {
      recommendations.push(`--kv-cache-dtype fp8 を設定すると、KVキャッシュの消費メモリを半減できます。`);
    }
    if (maxModelLen > 8192) {
      recommendations.push(`コンテキスト長（--max-model-len）を用途に応じた適正値（例: 4096 や 8192）に下げることを検討してください。`);
    }
  } else if (maxConcurrencyAtMaxLen < 4.0) {
    warnings.push(`KVキャッシュ枠は確保できていますが、最大コンテキスト（${maxModelLen}）時の同時リクエスト数は約 ${maxConcurrencyAtMaxLen.toFixed(1)} req です。`);
  }

  // Check TP compatibility with KV heads
  if (gpuCount > 1) {
    if (kvHeads % gpuCount !== 0) {
      warnings.push(`KVヘッド数（${kvHeads}）がGPU枚数 / TP（${gpuCount}）で割り切れません。vLLMが起動エラーになる可能性があります（TPは ${getDivisors(kvHeads).join(', ')} などを推奨）。`);
    }
    if (gpuCount > kvHeads) {
      errors.push(`TP数（${gpuCount}）がKVヘッド数（${kvHeads}）を超えています。vLLMでは TP <= KVヘッド数 である必要があります。`);
    }
  }

  // Optimization tips
  if (enablePrefixCaching) {
    recommendations.push(`--enable-prefix-caching が有効です。システムプロンプトやマルチターン会話で高速化とメモリ再利用が期待できます。`);
  } else {
    recommendations.push(`チャットやRAG用途の場合は --enable-prefix-caching の有効化をお勧めします。`);
  }

  return {
    status,
    totalVramPerGpu,
    usableVramPerGpu,
    weightWithOverheadPerGpuGb,
    weightPerGpuGb,
    totalModelWeightGb,
    cudaRuntimeGb,
    cudaGraphGb,
    baseOverheadPerGpuGb,
    vllmRemainingForKvGb: Math.max(0, vllmRemainingForKvGb),
    safetyMarginGb,
    kvKbPerTokenPerGpu,
    maxKvTokens,
    maxConcurrencyAtMaxLen,
    isDirectFileMode,
    warnings,
    recommendations,
    errors,
  };
}

function getDivisors(n) {
  const divs = [];
  for (let i = 1; i <= n; i++) {
    if (n % i === 0) divs.push(i);
  }
  return divs;
}

export function generateCommands(config) {
  const {
    modelId,
    gpuCount,
    gpuMemoryUtilization,
    maxModelLen,
    quantizationId,
    kvCacheDtype,
    enforceEager,
    enablePrefixCaching,
    enableChunkedPrefill,
    maxNumBatchedTokens,
    maxNumSeqs,
    cpuOffloadGb,
    port,
    host,
    apiKey,
    servedModelName,
  } = config;

  const quantOption = QUANT_OPTIONS.find(q => q.id === quantizationId);

  // CLI Arguments array
  const args = [`vllm serve ${modelId}`];

  if (servedModelName && servedModelName.trim()) {
    args.push(`--served-model-name ${servedModelName.trim()}`);
  }

  if (gpuCount > 1) {
    args.push(`--tensor-parallel-size ${gpuCount}`);
  }

  if (gpuMemoryUtilization !== 0.9) {
    args.push(`--gpu-memory-utilization ${gpuMemoryUtilization.toFixed(2)}`);
  }

  if (maxModelLen) {
    args.push(`--max-model-len ${maxModelLen}`);
  }

  if (quantOption && quantOption.flag) {
    args.push(quantOption.flag);
  }

  if (kvCacheDtype === 'fp8') {
    args.push(`--kv-cache-dtype fp8`);
  }

  if (enforceEager) {
    args.push(`--enforce-eager`);
  }

  if (enablePrefixCaching) {
    args.push(`--enable-prefix-caching`);
  }

  if (enableChunkedPrefill) {
    args.push(`--enable-chunked-prefill`);
  }

  if (maxNumBatchedTokens && maxNumBatchedTokens !== 2048) {
    args.push(`--max-num-batched-tokens ${maxNumBatchedTokens}`);
  }

  if (maxNumSeqs && maxNumSeqs !== 256) {
    args.push(`--max-num-seqs ${maxNumSeqs}`);
  }

  if (cpuOffloadGb > 0) {
    args.push(`--cpu-offload-gb ${cpuOffloadGb}`);
  }

  if (host && host !== '0.0.0.0') {
    args.push(`--host ${host}`);
  }

  if (port && port !== 8000) {
    args.push(`--port ${port}`);
  }

  if (apiKey && apiKey.trim()) {
    args.push(`--api-key ${apiKey.trim()}`);
  }

  // 1. Single line CLI
  const cliSingleLine = args.join(' ');

  // 2. Multiline CLI (formatted with backslashes)
  const cliMultiLine = args.map((arg, idx) => {
    if (idx === 0) return arg;
    return `  ${arg}`;
  }).join(' \\\n');

  // 3. Docker Run Command
  const dockerArgs = [
    'docker run --runtime nvidia --gpus all',
    '-v ~/.cache/huggingface:/root/.cache/huggingface',
    '--env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"',
    `-p ${port || 8000}:${port || 8000}`,
    '--ipc=host',
    'vllm/vllm-openai:latest',
  ];
  const dockerServeArgs = args.slice(1);
  const dockerCmd = `${dockerArgs.join(' \\\n  ')} \\\n  ${modelId} \\\n  ${dockerServeArgs.join(' \\\n  ')}`;

  // 4. Docker Compose
  const composeYaml = `version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    runtime: nvidia
    restart: unless-stopped
    ports:
      - "${port || 8000}:${port || 8000}"
    environment:
      - HUGGING_FACE_HUB_TOKEN=\${HF_TOKEN}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: ${gpuCount === 1 ? 'all' : gpuCount}
              capabilities: [gpu]
    command: >
      ${modelId}
      ${dockerServeArgs.map(a => `      ${a}`).join('\n')}
`;

  // 5. Python vLLM Code
  const pythonCode = `from vllm import LLM, SamplingParams

# Initialize vLLM Offline Engine / Server
llm = LLM(
    model="${modelId}",
    tensor_parallel_size=${gpuCount},
    gpu_memory_utilization=${gpuMemoryUtilization},
    max_model_len=${maxModelLen},
    ${quantOption && quantOption.id !== 'bfloat16' ? `quantization="${quantOption.id.startsWith('modelopt') ? 'modelopt' : quantOption.id.startsWith('bitsandbytes') ? 'bitsandbytes' : quantOption.id}",\n    ` : ''}${kvCacheDtype === 'fp8' ? `kv_cache_dtype="fp8",\n    ` : ''}${enforceEager ? `enforce_eager=True,\n    ` : ''}${enablePrefixCaching ? `enable_prefix_caching=True,\n    ` : ''}
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,
)

prompts = ["Hello! How does vLLM optimize GPU VRAM memory?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
`;

  return {
    cliSingleLine,
    cliMultiLine,
    dockerCmd,
    composeYaml,
    pythonCode,
  };
}
