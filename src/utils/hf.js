/**
 * Utility to parse HuggingFace URLs / Model IDs and fetch model configuration & weights metadata
 */

export function parseHfInput(input) {
  if (!input) return null;
  let clean = input.trim();
  clean = clean.replace(/\/+$/, '');

  const urlPattern = /huggingface\.co\/([^\/\s]+\/[^\/\s]+)/i;
  const match = clean.match(urlPattern);
  if (match && match[1]) {
    return match[1];
  }

  const repoPattern = /^([a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+)$/;
  if (repoPattern.test(clean)) {
    return clean;
  }

  return clean;
}

export async function fetchHfModelInfo(modelId, token = '') {
  const headers = {};
  if (token && token.trim()) {
    headers['Authorization'] = `Bearer ${token.trim()}`;
  }

  const result = {
    modelId,
    name: modelId.split('/')[1] || modelId,
    paramsB: null,
    layers: 32,
    hiddenSize: 4096,
    attnHeads: 32,
    kvHeads: 8,
    maxContext: 8192,
    defaultDtype: 'bfloat16',
    quantization: 'none',
    detectedQuantId: 'bfloat16',
    detectedKvCacheDtype: null,
    safetensorsSizeGb: null,
  };

  try {
    // 1. Fetch config.json
    const configUrl = `https://huggingface.co/${modelId}/raw/main/config.json`;
    const configRes = await fetch(configUrl, { headers });

    if (!configRes.ok) {
      if (configRes.status === 401 || configRes.status === 403) {
        throw new Error('モデルへのアクセスが制限されているか（Gated Model）、認証が必要です。HF Tokenを入力してください。');
      } else if (configRes.status === 404) {
        throw new Error(`モデル "${modelId}" が見つかりませんでした (404)`);
      } else {
        throw new Error(`config.json の取得に失敗しました (HTTP ${configRes.status})`);
      }
    }

    const config = await configRes.json();

    // Support multimodal models where text model is under text_config (e.g. GLM-5 Next, Qwen2-VL)
    const tCfg = config.text_config || config;

    // Architecture specs
    result.layers = tCfg.num_hidden_layers || tCfg.n_layer || tCfg.num_layers || 32;
    result.hiddenSize = tCfg.hidden_size || tCfg.n_embd || 4096;
    result.attnHeads = tCfg.num_attention_heads || tCfg.n_head || 32;
    result.kvHeads = tCfg.num_key_value_heads || tCfg.num_kv_heads || result.attnHeads;
    result.headDim = tCfg.head_dim || config.head_dim || Math.round(result.hiddenSize / result.attnHeads);

    // Max context length
    result.maxContext = tCfg.max_position_embeddings || 
                        tCfg.max_sequence_length || 
                        tCfg.seq_length || 
                        tCfg.sliding_window || 
                        config.max_position_embeddings ||
                        8192;

    // Base precision
    const rawDtype = tCfg.torch_dtype || config.torch_dtype || tCfg.dtype || config.dtype;
    if (rawDtype) {
      result.defaultDtype = String(rawDtype).replace('torch.', '');
    }

    // Check Hybrid Linear Attention / Sparse Attention (e.g. GLM-5 Next / GLM-5.3-Flash)
    let fullAttnLayers = result.layers;
    let isHybridLinear = false;

    if (tCfg.layer_types && Array.isArray(tCfg.layer_types)) {
      const fullLayers = tCfg.layer_types.filter(t => 
        t.includes('full') || t.includes('sparse') || (t.includes('attention') && !t.includes('linear'))
      );
      if (fullLayers.length > 0 && fullLayers.length < tCfg.layer_types.length) {
        fullAttnLayers = fullLayers.length;
        isHybridLinear = true;
      }
    } else if (tCfg.linear_attn_config && tCfg.linear_attn_config.full_attn_layers) {
      fullAttnLayers = tCfg.linear_attn_config.full_attn_layers.length;
      isHybridLinear = true;
    }

    // MLA (Multi-Head Latent Attention) detection (DeepSeek V2/V3/R1, GLM-5 Next, etc.)
    const isMla = Boolean(
      tCfg.kv_lora_rank || 
      (tCfg.model_type && (tCfg.model_type.includes('deepseek') || tCfg.model_type.includes('glm5'))) ||
      (config.model_type && (config.model_type.includes('deepseek') || config.model_type.includes('glm5')))
    );
    result.isMla = isMla;
    result.isHybridLinear = isHybridLinear;
    result.fullAttnLayers = fullAttnLayers;

    if (isMla) {
      result.kvLoraRank = tCfg.kv_lora_rank || 512;
      result.qkRopeHeadDim = tCfg.qk_rope_head_dim || 0;
    }

    // 2. Try fetching hf_quant_config.json (Used by NVIDIA ModelOpt / NVFP4 / TensorRT-LLM)
    try {
      const hfQuantUrl = `https://huggingface.co/${modelId}/raw/main/hf_quant_config.json`;
      const hfQuantRes = await fetch(hfQuantUrl, { headers });
      if (hfQuantRes.ok) {
        const quantData = await hfQuantRes.json();
        const algo = (quantData.quantization?.quant_algo || '').toUpperCase();
        if (algo === 'NVFP4' || algo.includes('FP4')) {
          result.detectedQuantId = 'modelopt_nvfp4';
          result.quantization = 'NVFP4 (ModelOpt)';
        } else if (algo === 'FP8') {
          result.detectedQuantId = 'fp8';
          result.quantization = 'FP8';
        }

        if (quantData.quantization?.kv_cache_quant_algo === 'FP8') {
          result.detectedKvCacheDtype = 'fp8';
        }
      }
    } catch (e) {
      // Non-fatal
    }

    // 3. Inspect config.quantization_config if not already detected
    if (result.detectedQuantId === 'bfloat16' && config.quantization_config) {
      const q = config.quantization_config;
      const method = (q.quant_method || '').toLowerCase();
      const algo = (q.quant_algo || '').toUpperCase();

      if (algo === 'NVFP4' || method === 'modelopt' || (q.quant_algo && q.quant_algo.includes('FP4'))) {
        result.detectedQuantId = 'modelopt_nvfp4';
        result.quantization = 'NVFP4 (ModelOpt)';
      } else if (method === 'fp8' || method.includes('fp8') || algo === 'FP8') {
        result.detectedQuantId = 'fp8';
        result.quantization = 'FP8';
      } else if (method === 'awq') {
        result.detectedQuantId = 'awq';
        result.quantization = 'AWQ 4-bit';
      } else if (method === 'gptq') {
        result.detectedQuantId = 'gptq';
        result.quantization = 'GPTQ 4-bit';
      } else if (method === 'compressed-tensors') {
        result.detectedQuantId = 'compressed-tensors';
        result.quantization = 'Compressed Tensors';
      } else if (method === 'bitsandbytes') {
        result.detectedQuantId = q.bits === 8 ? 'bitsandbytes_8bit' : 'bitsandbytes';
        result.quantization = `BitsAndBytes ${q.bits || 4}-bit`;
      } else if (method === 'marlin') {
        result.detectedQuantId = 'marlin';
        result.quantization = 'Marlin 4-bit';
      }

      // Check if kv_cache_scheme is defined (e.g. llm-compressor / compressed-tensors FP8 KV cache)
      if (q.kv_cache_scheme && q.kv_cache_scheme.num_bits === 8) {
        result.detectedKvCacheDtype = 'fp8';
      }
    }

    // 4. Fetch HF API models metadata with ?blobs=true for total checkpoint disk size
    let tags = [];
    try {
      const apiModelUrl = `https://huggingface.co/api/models/${modelId}?blobs=true`;
      const apiRes = await fetch(apiModelUrl, { headers });
      if (apiRes.ok) {
        const apiData = await apiRes.json();
        tags = apiData.tags || [];

        // Calculate total safetensors size
        const safetensorsFiles = (apiData.siblings || []).filter(f => 
          (f.rfilename || '').endsWith('.safetensors')
        );
        const totalSafetensorsBytes = safetensorsFiles.reduce((acc, f) => acc + (f.size || 0), 0);
        if (totalSafetensorsBytes > 0) {
          result.safetensorsSizeGb = Number((totalSafetensorsBytes / (1024 ** 3)).toFixed(1));
        }

        // Check parameter counts
        if (apiData.safetensors && apiData.safetensors.total) {
          const params = apiData.safetensors.parameters || {};

          // In NVFP4, weights are often packed into U8 (uint8, 2 params/byte)
          if (params.U8 && params.U8 > 1e10) {
            // Packed 4-bit params
            const estimated4BitParams = params.U8 * 2;
            const otherParams = (params.BF16 || 0) + (params.F8_E4M3 || 0) + (params.F32 || 0);
            result.paramsB = Number(((estimated4BitParams + otherParams) / 1e9).toFixed(1));
            result.detectedQuantId = 'modelopt_nvfp4';
            result.quantization = 'NVFP4 (U8 packed)';
          } else {
            result.paramsB = Number((apiData.safetensors.total / 1e9).toFixed(2));
          }
        }
      }
    } catch (e) {
      console.warn('Hugging Face API blobs fetch error:', e);
    }

    // 5. Name / Tag Heuristics if still unresolved
    const lowerId = modelId.toLowerCase();
    const lowerTags = tags.map(t => String(t).toLowerCase());

    if (result.detectedQuantId === 'bfloat16') {
      if (lowerId.includes('nvfp4') || lowerTags.includes('nvfp4') || lowerTags.includes('fp4') || lowerId.includes('-fp4')) {
        result.detectedQuantId = 'modelopt_nvfp4';
        result.quantization = 'NVFP4 (タグ/モデル名より検出)';
      } else if (lowerId.includes('awq') || lowerTags.includes('awq')) {
        result.detectedQuantId = 'awq';
        result.quantization = 'AWQ (タグ/モデル名より検出)';
      } else if (lowerId.includes('gptq') || lowerTags.includes('gptq')) {
        result.detectedQuantId = 'gptq';
        result.quantization = 'GPTQ (タグ/モデル名より検出)';
      } else if (lowerId.includes('fp8') || lowerTags.includes('fp8')) {
        result.detectedQuantId = 'fp8';
        result.quantization = 'FP8 (タグ/モデル名より検出)';
      } else if (lowerId.includes('bnb') || lowerId.includes('bitsandbytes')) {
        result.detectedQuantId = 'bitsandbytes';
        result.quantization = 'BitsAndBytes';
      }
    }

    // 6. Checkpoint File Size vs Parameter Count verification
    if (result.safetensorsSizeGb && result.paramsB) {
      const bytesPerParamEst = (result.safetensorsSizeGb * (1024 ** 3)) / (result.paramsB * 1e9);
      if (result.detectedQuantId === 'bfloat16' && bytesPerParamEst < 0.75) {
        // Definitely 4-bit!
        result.detectedQuantId = lowerId.includes('nvfp4') ? 'modelopt_nvfp4' : 'awq';
        result.quantization = `${result.detectedQuantId.toUpperCase()} (ファイル実サイズ ${result.safetensorsSizeGb}GB より推定)`;
      } else if (result.detectedQuantId === 'bfloat16' && bytesPerParamEst < 1.35) {
        result.detectedQuantId = 'fp8';
        result.quantization = `FP8 (ファイル実サイズ ${result.safetensorsSizeGb}GB より推定)`;
      }
    }

    // 7. If parameter count still null, fallback to estimate
    if (!result.paramsB) {
      const vocabSize = config.vocab_size || 32000;
      const interSize = config.intermediate_size || (config.hidden_size * 4);
      const isGated = config.hidden_act && (config.hidden_act.includes('glu') || config.hidden_act.includes('silu'));
      const mlpParams = isGated ? 3 * result.hiddenSize * interSize : 2 * result.hiddenSize * interSize;
      const attnParams = (result.hiddenSize * result.hiddenSize) + 
                         (2 * result.hiddenSize * (result.hiddenSize / result.attnHeads) * result.kvHeads) + 
                         (result.hiddenSize * result.hiddenSize);
      const layerParams = (attnParams + mlpParams) * result.layers;
      const embedParams = vocabSize * result.hiddenSize;
      const totalEstimated = layerParams + embedParams;
      result.paramsB = Number((totalEstimated / 1e9).toFixed(1));
    }

    return result;
  } catch (err) {
    throw err;
  }
}
