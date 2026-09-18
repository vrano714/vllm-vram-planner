import React, { useState } from 'react';
import { Box, DownloadCloud, Key, CheckCircle, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { QUANT_OPTIONS } from '../data/presets';
import { parseHfInput, fetchHfModelInfo } from '../utils/hf';

export default function ModelConfig({
  modelConfig,
  setModelConfig,
  onModelLoaded,
}) {
  const [hfInput, setHfInput] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [showTokenInput, setShowTokenInput] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);
  const [fetchSuccess, setFetchSuccess] = useState(null);
  const [showAdvancedParams, setShowAdvancedParams] = useState(false);

  // Handle Hugging Face Fetch
  const handleFetchHf = async (e) => {
    if (e) e.preventDefault();
    const parsedId = parseHfInput(hfInput);
    if (!parsedId) {
      setFetchError('有効なHuggingFace URLまたはモデルID（例: Qwen/Qwen2.5-7B-Instruct）を入力してください。');
      return;
    }

    setIsLoading(true);
    setFetchError(null);
    setFetchSuccess(null);

    try {
      const info = await fetchHfModelInfo(parsedId, hfToken);
      
      const updated = {
        ...modelConfig,
        modelId: info.modelId,
        paramsB: info.paramsB || 7.0,
        layers: info.layers,
        hiddenSize: info.hiddenSize,
        attnHeads: info.attnHeads,
        kvHeads: info.kvHeads,
        headDim: info.headDim || null,
        maxModelLen: Math.min(modelConfig.maxModelLen, info.maxContext) || info.maxContext || 8192,
        nativeMaxContext: info.maxContext,
        safetensorsSizeGb: info.safetensorsSizeGb || null,
        customFileSizeGb: info.safetensorsSizeGb || null,
        useDirectFileSize: Boolean(info.safetensorsSizeGb && (info.detectedQuantId === 'modelopt_nvfp4' || info.detectedQuantId !== 'bfloat16')),
        detectedKvCacheDtype: info.detectedKvCacheDtype || null,
        isMla: info.isMla || false,
        isHybridLinear: info.isHybridLinear || false,
        fullAttnLayers: info.fullAttnLayers || null,
        kvLoraRank: info.kvLoraRank || null,
        qkRopeHeadDim: info.qkRopeHeadDim || null,
      };

      // Set detected quantization
      if (info.detectedQuantId) {
        updated.quantizationId = info.detectedQuantId;
      }

      setModelConfig(updated);
      if (onModelLoaded) onModelLoaded(updated);

      const quantLabel = QUANT_OPTIONS.find(q => q.id === updated.quantizationId)?.name || updated.quantizationId;
      const sizeMsg = info.safetensorsSizeGb ? ` / ファイル実容量: 約${info.safetensorsSizeGb} GB` : '';
      setFetchSuccess(`「${info.name}」のスペック解析完了！（約 ${info.paramsB}B パラメータ / ${info.layers}層 / 精度: ${quantLabel}${sizeMsg}）`);
    } catch (err) {
      setFetchError(err.message || 'モデル情報の取得に失敗しました');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm dark:shadow-lg space-y-4 transition-colors">
      <div className="flex items-center justify-between pb-3 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-lg">
          <Box className="w-5 h-5" />
          <span>2. モデル設定 & Hugging Face 連動</span>
        </div>
        <span className="text-xs font-mono bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 px-2.5 py-0.5 rounded-full">
          {modelConfig.paramsB} B
        </span>
      </div>

      {/* Hugging Face URL Fetcher */}
      <div className="space-y-2">
        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
          Hugging Face URL / モデルリポジトリID からスペック自動読み込み
        </label>
        <form onSubmit={handleFetchHf} className="flex gap-2">
          <input
            type="text"
            placeholder="例: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct または Qwen/Qwen2.5-7B-Instruct"
            value={hfInput}
            onChange={(e) => setHfInput(e.target.value)}
            className="flex-1 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-xs font-mono text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:border-indigo-500"
          />
          <button
            type="submit"
            disabled={isLoading || !hfInput.trim()}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium transition flex items-center gap-1.5 whitespace-nowrap shadow-sm"
          >
            <DownloadCloud className={`w-4 h-4 ${isLoading ? 'animate-bounce' : ''}`} />
            {isLoading ? '解析中...' : 'スペック取得'}
          </button>
        </form>

        {/* HF Token toggle for gated models */}
        <div className="flex items-center text-[11px] text-gray-500 dark:text-gray-400 pt-0.5">
          <button
            type="button"
            onClick={() => setShowTokenInput(!showTokenInput)}
            className="hover:text-indigo-600 dark:hover:text-indigo-400 flex items-center gap-1 transition"
          >
            <Key className="w-3 h-3" />
            {showTokenInput ? 'HF Token 入力を閉じる' : 'Gated / 非公開モデル用 HF Token を指定する'}
          </button>
        </div>

        {showTokenInput && (
          <div className="bg-gray-50 dark:bg-gray-900/80 p-2.5 rounded-lg border border-gray-200 dark:border-gray-800 space-y-1 mt-1">
            <label className="text-[11px] text-gray-600 dark:text-gray-400 block">Hugging Face Read Token (hf_...)</label>
            <input
              type="password"
              placeholder="hf_xxxxxxxxxxxxxxxxxxxxxx"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1.5 text-xs font-mono text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 focus:outline-none focus:border-indigo-500"
            />
          </div>
        )}

        {/* Fetch Status Messages */}
        {fetchSuccess && (
          <div className="p-2.5 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-300 dark:border-emerald-800/80 rounded-lg text-xs text-emerald-800 dark:text-emerald-300 flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
            <span>{fetchSuccess}</span>
          </div>
        )}

        {fetchError && (
          <div className="p-2.5 bg-rose-50 dark:bg-rose-950/40 border border-rose-300 dark:border-rose-800/80 rounded-lg text-xs text-rose-800 dark:text-rose-300 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0 mt-0.5" />
            <span>{fetchError}</span>
          </div>
        )}
      </div>

      {/* Quantization / Precision */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80 space-y-2.5">
        <div className="flex justify-between items-center text-xs">
          <label className="text-gray-700 dark:text-gray-300 font-medium">量子化・サービング設定 (Quantization / dtype)</label>
          <span className="text-xs text-indigo-600 dark:text-indigo-400 font-mono">
            {QUANT_OPTIONS.find(q => q.id === modelConfig.quantizationId)?.bytesPerParam} Bytes / param
          </span>
        </div>
        <select
          value={modelConfig.quantizationId}
          onChange={(e) => setModelConfig({ ...modelConfig, quantizationId: e.target.value })}
          className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-900 dark:text-white focus:outline-none focus:border-indigo-500 font-mono"
        >
          {QUANT_OPTIONS.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.name} ({opt.bytesPerParam} bytes/param) {opt.flag ? `[${opt.flag}]` : ''}
            </option>
          ))}
        </select>

        {/* Option: Use Direct File Size on VRAM */}
        <div className={`p-3 rounded-lg border transition ${
          modelConfig.useDirectFileSize 
            ? 'bg-indigo-50/80 dark:bg-indigo-950/40 border-indigo-300 dark:border-indigo-500/80' 
            : 'bg-gray-50 dark:bg-gray-900/60 border-gray-200 dark:border-gray-800'
        }`}>
          <label className="flex items-start gap-2.5 cursor-pointer">
            <input
              type="checkbox"
              checked={modelConfig.useDirectFileSize || false}
              onChange={(e) => setModelConfig({ 
                ...modelConfig, 
                useDirectFileSize: e.target.checked,
                customFileSizeGb: modelConfig.customFileSizeGb || modelConfig.safetensorsSizeGb || 0
              })}
              className="mt-0.5 w-4 h-4 rounded text-indigo-600 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 focus:ring-0"
            />
            <div className="text-xs flex-1">
              <div className="flex items-center gap-2 font-semibold text-gray-900 dark:text-white">
                <span>実ファイル容量をそのままVRAMに乗せる</span>
                <span className="bg-indigo-100 dark:bg-indigo-900/80 text-indigo-700 dark:text-indigo-300 text-[10px] px-1.5 py-0.5 rounded font-mono border border-indigo-200 dark:border-indigo-700/60">
                  {modelConfig.useDirectFileSize ? '有効' : 'オフ'}
                </span>
              </div>
              <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                パラメータ数からの理論計算をバイパスし、safetensors実サイズをそのままGPU枚数（TP）で等分してVRAMに割り当てます（NVFP4やMoE等で正確）。
              </p>
            </div>
          </label>

          {modelConfig.useDirectFileSize && (
            <div className="mt-2.5 pt-2 border-t border-indigo-200 dark:border-indigo-900/60 flex items-center justify-between text-xs">
              <span className="text-gray-700 dark:text-gray-300">実ファイル総容量 (safetensors合計):</span>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  max="10000"
                  value={modelConfig.customFileSizeGb || modelConfig.safetensorsSizeGb || 0}
                  onChange={(e) => setModelConfig({ 
                    ...modelConfig, 
                    customFileSizeGb: parseFloat(e.target.value) || 0 
                  })}
                  className="w-24 bg-white dark:bg-gray-800 border border-indigo-400 dark:border-indigo-500 rounded px-2.5 py-1 text-right font-mono text-xs text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                <span className="text-gray-500 dark:text-gray-400 font-mono">GB</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Fine-tuning parameters detail toggle */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80">
        <button
          type="button"
          onClick={() => setShowAdvancedParams(!showAdvancedParams)}
          className="text-xs text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 flex items-center justify-between w-full py-1 transition"
        >
          <span>アーキテクチャ詳細・パラメータ手動微調整 (KVヘッド/層数など)</span>
          {showAdvancedParams ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {showAdvancedParams && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 mt-2 bg-gray-50 dark:bg-gray-900/60 p-3 rounded-lg border border-gray-200 dark:border-gray-800 text-xs">
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">パラメータ数 (B)</label>
              <input
                type="number"
                step="0.1"
                min="0.5"
                max="1000"
                value={modelConfig.paramsB}
                onChange={(e) => setModelConfig({ ...modelConfig, paramsB: parseFloat(e.target.value) || 7.0 })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">全レイヤー数 (Layers)</label>
              <input
                type="number"
                min="1"
                max="200"
                value={modelConfig.layers}
                onChange={(e) => setModelConfig({ ...modelConfig, layers: parseInt(e.target.value, 10) || 32 })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">隠れ層 (Hidden Size)</label>
              <input
                type="number"
                min="512"
                max="16384"
                step="64"
                value={modelConfig.hiddenSize}
                onChange={(e) => setModelConfig({ ...modelConfig, hiddenSize: parseInt(e.target.value, 10) || 4096 })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">アテンションヘッド数</label>
              <input
                type="number"
                min="1"
                max="128"
                value={modelConfig.attnHeads}
                onChange={(e) => setModelConfig({ ...modelConfig, attnHeads: parseInt(e.target.value, 10) || 32 })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">KVヘッド数 (GQA)</label>
              <input
                type="number"
                min="1"
                max="128"
                value={modelConfig.kvHeads}
                onChange={(e) => setModelConfig({ ...modelConfig, kvHeads: parseInt(e.target.value, 10) || 8 })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">ネイティブ最大長</label>
              <input
                type="number"
                step="1024"
                value={modelConfig.nativeMaxContext || 32768}
                onChange={(e) => setModelConfig({ ...modelConfig, nativeMaxContext: parseInt(e.target.value, 10) || 32768 })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">Head Dim (ヘッド次元数)</label>
              <input
                type="number"
                min="16"
                max="512"
                step="8"
                placeholder={modelConfig.attnHeads ? String(Math.round(modelConfig.hiddenSize / modelConfig.attnHeads)) : '128'}
                value={modelConfig.headDim || ''}
                onChange={(e) => setModelConfig({ ...modelConfig, headDim: e.target.value ? parseInt(e.target.value, 10) : null })}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
