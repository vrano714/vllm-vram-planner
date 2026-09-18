export const DEFAULT_MODEL_CONFIG = {
  id: 'Qwen/Qwen2.5-7B-Instruct',
  name: 'Qwen 2.5 7B Instruct',
  paramsB: 7.61,
  layers: 28,
  hiddenSize: 3584,
  attnHeads: 28,
  kvHeads: 4,
  maxContext: 32768,
  defaultDtype: 'bfloat16',
};


export const QUANT_OPTIONS = [
  { id: 'bfloat16', name: 'BF16 / FP16 (16-bit)', bytesPerParam: 2.0, flag: '' },
  { id: 'modelopt_nvfp4', name: 'NVFP4 / FP4 (4-bit NVIDIA ModelOpt)', bytesPerParam: 0.55, flag: '--quantization modelopt' },
  { id: 'fp8', name: 'FP8 (8-bit e4m3/e5m2)', bytesPerParam: 1.0, flag: '--quantization fp8' },
  { id: 'awq', name: 'AWQ 4-bit', bytesPerParam: 0.55, flag: '--quantization awq' },
  { id: 'gptq', name: 'GPTQ 4-bit', bytesPerParam: 0.55, flag: '--quantization gptq' },
  { id: 'compressed-tensors', name: 'Compressed Tensors (W4A16 / FP8)', bytesPerParam: 0.55, flag: '--quantization compressed-tensors' },
  { id: 'bitsandbytes', name: 'BitsAndBytes 4-bit', bytesPerParam: 0.55, flag: '--quantization bitsandbytes --load-format bitsandbytes' },
  { id: 'bitsandbytes_8bit', name: 'BitsAndBytes 8-bit', bytesPerParam: 1.05, flag: '--quantization bitsandbytes --load-format bitsandbytes' },
  { id: 'marlin', name: 'Marlin (Optimized 4-bit)', bytesPerParam: 0.55, flag: '--quantization marlin' },
];
