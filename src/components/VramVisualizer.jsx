import React from 'react';
import { Activity, AlertOctagon, AlertTriangle, CheckCircle2, Lightbulb, HardDrive, MessageSquare, Zap } from 'lucide-react';

export default function VramVisualizer({ calc, config }) {
  const {
    status,
    totalVramPerGpu,
    usableVramPerGpu,
    weightWithOverheadPerGpuGb,
    cudaRuntimeGb,
    cudaGraphGb,
    vllmRemainingForKvGb,
    safetyMarginGb,
    kvKbPerTokenPerGpu,
    maxKvTokens,
    maxConcurrencyAtMaxLen,
    isDirectFileMode,
    totalModelWeightGb,
    warnings,
    recommendations,
    errors,
  } = calc;

  // Percentage calculations for stacked bar
  const pctCudaRuntime = (cudaRuntimeGb / totalVramPerGpu) * 100;
  const pctCudaGraph = (cudaGraphGb / totalVramPerGpu) * 100;
  const pctWeights = (weightWithOverheadPerGpuGb / totalVramPerGpu) * 100;
  const pctKv = Math.max(0, (vllmRemainingForKvGb / totalVramPerGpu) * 100);
  const pctMargin = (safetyMarginGb / totalVramPerGpu) * 100;

  const isOom = status === 'oom';

  return (
    <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm dark:shadow-lg space-y-5 transition-colors">
      {/* Header & Status Indicator */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-lg">
          <Activity className="w-5 h-5" />
          <span>4. VRAM メモリ配分 & スループット診断 (GPU 1基あたり)</span>
        </div>

        {/* Status Badge */}
        <div>
          {status === 'optimal' && (
            <span className="flex items-center gap-1.5 bg-emerald-50 dark:bg-emerald-950/70 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700/80 px-3 py-1 rounded-full text-xs font-semibold">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              サービング可能 (十分な余力)
            </span>
          )}
          {status === 'tight' && (
            <span className="flex items-center gap-1.5 bg-amber-50 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-700/80 px-3 py-1 rounded-full text-xs font-semibold">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
              VRAM逼迫 (チューニング推奨)
            </span>
          )}
          {status === 'oom' && (
            <span className="flex items-center gap-1.5 bg-rose-50 dark:bg-rose-950/70 text-rose-800 dark:text-rose-300 border border-rose-300 dark:border-rose-700/80 px-3 py-1 rounded-full text-xs font-semibold">
              <AlertOctagon className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400" />
              OOM (起動不可)
            </span>
          )}
        </div>
      </div>

      {/* Visual Stacked Memory Bar */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-700 dark:text-gray-300 font-medium">GPU VRAM 占有率 (全 {totalVramPerGpu} GB)</span>
          <span className="font-mono text-gray-500 dark:text-gray-400">
            利用枠: <strong className="text-gray-900 dark:text-white">{usableVramPerGpu.toFixed(1)} GB</strong> ({(config.gpuMemoryUtilization * 100).toFixed(0)}%)
          </span>
        </div>

        {/* The Bar */}
        <div className="h-9 w-full bg-gray-200 dark:bg-gray-900 rounded-lg overflow-hidden flex border border-gray-300 dark:border-gray-700/80 shadow-inner relative">
          {/* CUDA Runtime */}
          <div
            style={{ width: `${Math.min(pctCudaRuntime, 100)}%` }}
            className="bg-purple-600 h-full flex items-center justify-center text-[10px] text-white font-mono font-medium truncate px-1 transition-all duration-300"
            title={`CUDA & PyTorch 基盤: ${cudaRuntimeGb.toFixed(1)} GB`}
          >
            {pctCudaRuntime > 6 ? `${cudaRuntimeGb.toFixed(1)}G` : ''}
          </div>

          {/* CUDA Graph (if not eager) */}
          {cudaGraphGb > 0 && (
            <div
              style={{ width: `${Math.min(pctCudaGraph, 100)}%` }}
              className="bg-indigo-500 h-full flex items-center justify-center text-[10px] text-white font-mono font-medium truncate px-1 transition-all duration-300 border-l border-purple-800"
              title={`CUDA Graph キャプチャ: ${cudaGraphGb.toFixed(1)} GB`}
            >
              {pctCudaGraph > 6 ? `${cudaGraphGb.toFixed(1)}G` : ''}
            </div>
          )}

          {/* Model Weights */}
          <div
            style={{ width: `${Math.min(pctWeights, 100)}%` }}
            className="bg-blue-600 h-full flex items-center justify-center text-[11px] text-white font-mono font-semibold truncate px-1 transition-all duration-300 border-l border-indigo-700"
            title={`モデル重み: ${weightWithOverheadPerGpuGb.toFixed(2)} GB ${isDirectFileMode ? `(実ファイル ${totalModelWeightGb} GB ÷ ${config.gpuCount} GPU)` : ''}`}
          >
            {pctWeights > 10 ? `重み ${weightWithOverheadPerGpuGb.toFixed(1)} GB` : ''}
          </div>

          {/* KV Cache or OOM Overflow */}
          {!isOom ? (
            <div
              style={{ width: `${Math.min(pctKv, 100)}%` }}
              className="bg-emerald-500 h-full flex items-center justify-center text-[11px] text-emerald-950 font-mono font-bold truncate px-1 transition-all duration-300 border-l border-blue-700"
              title={`KV Cache 割当領域: ${vllmRemainingForKvGb.toFixed(2)} GB`}
            >
              {pctKv > 12 ? `KV Cache ${vllmRemainingForKvGb.toFixed(1)} GB` : ''}
            </div>
          ) : (
            <div
              className="flex-1 bg-rose-600 h-full flex items-center justify-center text-[11px] text-white font-bold animate-pulse"
              title="VRAMオーバーフロー"
            >
              容量超過 (OOM)
            </div>
          )}

          {/* Safety Margin (Headroom) */}
          {!isOom && pctMargin > 0 && (
            <div
              style={{ width: `${pctMargin}%` }}
              className="bg-gray-300/80 dark:bg-gray-800/80 h-full flex items-center justify-center text-[9px] text-gray-700 dark:text-gray-400 font-mono truncate px-1 transition-all duration-300 border-l border-dashed border-gray-400 dark:border-gray-600"
              title={`未割当・セーフティマージン: ${safetyMarginGb.toFixed(1)} GB`}
            >
              {pctMargin > 6 ? 'マージン' : ''}
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-3 pt-1 text-[11px] text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-purple-600"></span>
            <span>CUDA基盤 ({cudaRuntimeGb.toFixed(1)}GB)</span>
          </div>
          {cudaGraphGb > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-indigo-500"></span>
              <span>CUDA Graph ({cudaGraphGb.toFixed(1)}GB)</span>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-blue-600"></span>
            <span>モデル重み ({weightWithOverheadPerGpuGb.toFixed(1)}GB{isDirectFileMode ? ' [実ファイル容量]' : ''})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500"></span>
            <span className="text-emerald-600 dark:text-emerald-400 font-medium">KV Cache枠 ({vllmRemainingForKvGb.toFixed(1)}GB)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-gray-400 dark:bg-gray-700"></span>
            <span>予備マージン ({safetyMarginGb.toFixed(1)}GB)</span>
          </div>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-gray-50 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-800 p-3 rounded-lg">
          <div className="text-[11px] text-gray-500 dark:text-gray-400 flex items-center gap-1 mb-1">
            <HardDrive className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
            <span>KV Cache 容量</span>
          </div>
          <div className={`text-lg font-mono font-bold ${vllmRemainingForKvGb > 2 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>
            {vllmRemainingForKvGb.toFixed(2)} <span className="text-xs font-normal text-gray-500 dark:text-gray-400">GB</span>
          </div>
          <div className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">GPU 1基あたり</div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-800 p-3 rounded-lg">
          <div className="text-[11px] text-gray-500 dark:text-gray-400 flex items-center gap-1 mb-1">
            <Zap className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
            <span>1トークンあたりKV</span>
          </div>
          <div className="text-lg font-mono font-bold text-gray-900 dark:text-white">
            {kvKbPerTokenPerGpu.toFixed(1)} <span className="text-xs font-normal text-gray-500 dark:text-gray-400">KB</span>
          </div>
          <div className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">
            {config.kvCacheDtype === 'fp8' ? 'FP8 圧縮' : '16-bit 標準'}
            {config.isHybridLinear ? ' (ハイブリッド線形)' : config.isMla ? ' (MLA圧縮)' : ''}
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-800 p-3 rounded-lg">
          <div className="text-[11px] text-gray-500 dark:text-gray-400 flex items-center gap-1 mb-1">
            <MessageSquare className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
            <span>最大KVトークン枠</span>
          </div>
          <div className="text-lg font-mono font-bold text-gray-900 dark:text-white truncate">
            {maxKvTokens > 0 ? maxKvTokens.toLocaleString() : '0'}
          </div>
          <div className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">システム全体</div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-800 p-3 rounded-lg">
          <div className="text-[11px] text-gray-500 dark:text-gray-400 flex items-center gap-1 mb-1">
            <Activity className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
            <span>最大長時 同時処理数</span>
          </div>
          <div className={`text-lg font-mono font-bold ${maxConcurrencyAtMaxLen >= 2 ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>
            {maxConcurrencyAtMaxLen >= 1 
              ? maxConcurrencyAtMaxLen.toFixed(1)
              : maxConcurrencyAtMaxLen > 0.01 
                ? maxConcurrencyAtMaxLen.toFixed(2)
                : maxConcurrencyAtMaxLen > 0 
                  ? '< 0.01' 
                  : '0.0'} <span className="text-xs font-normal text-gray-500 dark:text-gray-400">req</span>
          </div>
          <div className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">@{config.maxModelLen.toLocaleString()} tok時</div>
        </div>
      </div>

      {/* Diagnostics / Advice Section */}
      {(errors.length > 0 || warnings.length > 0 || recommendations.length > 0) && (
        <div className="space-y-2 pt-2 border-t border-gray-200 dark:border-gray-800/80 text-xs">
          {/* Errors */}
          {errors.map((err, idx) => (
            <div key={idx} className="p-2.5 bg-rose-50 dark:bg-rose-950/40 border border-rose-300 dark:border-rose-800/80 rounded-lg text-rose-900 dark:text-rose-300 flex items-start gap-2">
              <AlertOctagon className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0 mt-0.5" />
              <span>{err}</span>
            </div>
          ))}

          {/* Warnings */}
          {warnings.map((warn, idx) => (
            <div key={idx} className="p-2.5 bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800/80 rounded-lg text-amber-900 dark:text-amber-300 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
              <span>{warn}</span>
            </div>
          ))}

          {/* Recommendations / Tips */}
          {recommendations.map((rec, idx) => (
            <div key={idx} className="p-2.5 bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800/60 text-indigo-900 dark:text-indigo-200 flex items-start gap-2">
              <Lightbulb className="w-4 h-4 text-indigo-600 dark:text-indigo-400 flex-shrink-0 mt-0.5" />
              <span>{rec}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
