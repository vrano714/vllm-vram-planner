import React from 'react';
import { Cpu, Layers } from 'lucide-react';
const VRAM_PRESET_BUTTONS = [8, 12, 16, 24, 32, 48, 96, 188];

export default function GpuConfig({
  gpuVramGb,
  setGpuVramGb,
  gpuCount,
  setGpuCount,
  gpuMemoryUtilization,
  setGpuMemoryUtilization,
}) {
  const handleVramChange = (val) => {
    setGpuVramGb(Math.max(1, Number(val) || 1));
  };

  return (
    <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm dark:shadow-lg space-y-4 transition-colors">
      <div className="flex items-center justify-between pb-3 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-lg">
          <Cpu className="w-5 h-5" />
          <span>1. GPU & VRAM 構成</span>
        </div>
        <div className="text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 px-2.5 py-1 rounded-full border border-gray-300 dark:border-gray-700">
          合計 VRAM: <strong className="text-gray-900 dark:text-white font-mono">{gpuVramGb * gpuCount} GB</strong> ({gpuVramGb} GB × {gpuCount}基)
        </div>
      </div>

      {/* VRAM per GPU: Buttons + Slider + Direct Input */}
      <div className="space-y-3">
        <div className="flex justify-between items-center text-xs">
          <label className="text-gray-700 dark:text-gray-300 font-medium">1枚あたりの VRAM 容量 (GB)</label>
          <span className="font-mono text-indigo-600 dark:text-indigo-400 font-semibold text-sm">
            {gpuVramGb} GB
          </span>
        </div>

        {/* Quick Capacity Buttons */}
        <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5">
          {VRAM_PRESET_BUTTONS.map((gb) => {
            const active = gpuVramGb === gb;
            return (
              <button
                key={gb}
                type="button"
                onClick={() => setGpuVramGb(gb)}
                className={`py-1.5 rounded-lg text-xs font-mono font-medium border transition ${
                  active
                    ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm ring-1 ring-indigo-500/50'
                    : 'bg-gray-50 dark:bg-gray-800/60 border-gray-200 dark:border-gray-700/60 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {gb}GB
              </button>
            );
          })}
        </div>

        {/* Slider & Direct Number Input */}
        <div className="pt-1">
          <div className="flex items-center gap-3">
            <input
              type="range"
              min="4"
              max="384"
              step="1"
              value={gpuVramGb}
              onChange={(e) => handleVramChange(e.target.value)}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex items-center gap-1.5 flex-shrink-0">
              <input
                type="number"
                min="1"
                max="1024"
                value={gpuVramGb}
                onChange={(e) => handleVramChange(e.target.value)}
                className="w-20 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg px-2.5 py-1.5 text-center font-mono text-sm text-gray-900 dark:text-white focus:outline-none focus:border-indigo-500"
              />
              <span className="text-xs text-gray-500 font-mono">GB</span>
            </div>
          </div>
          <div className="flex justify-between text-[10px] text-gray-500 dark:text-gray-400 font-mono mt-1">
            <span>4 GB</span>
            <span>96 GB</span>
            <span>188 GB</span>
            <span>384 GB (最大)</span>
          </div>
        </div>
      </div>

      {/* GPU Count (Tensor Parallel) */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80">
        <div className="flex justify-between items-center text-xs mb-2">
          <label className="text-gray-700 dark:text-gray-300 font-medium flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
            GPU 枚数 (Tensor Parallel / TP サイズ)
          </label>
          <span className="font-mono text-indigo-600 dark:text-indigo-400 font-semibold">{gpuCount} 基</span>
        </div>
        <div className="grid grid-cols-6 gap-1.5">
          {[1, 2, 4, 8, 16, 32].map((count) => (
            <button
              key={count}
              type="button"
              onClick={() => setGpuCount(count)}
              className={`py-1.5 rounded text-xs font-mono font-medium border transition ${
                gpuCount === count
                  ? 'bg-indigo-600 border-indigo-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-800/60 border-gray-200 dark:border-gray-700/60 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              {count}
            </button>
          ))}
        </div>
      </div>

      {/* GPU Memory Utilization */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80">
        <div className="flex justify-between items-center text-xs mb-1.5">
          <label className="text-gray-700 dark:text-gray-300 font-medium">
            --gpu-memory-utilization (GPU利用枠)
          </label>
          <span className="font-mono text-indigo-600 dark:text-indigo-400 font-semibold">
            {(gpuMemoryUtilization * 100).toFixed(0)}% ({(gpuVramGb * gpuMemoryUtilization).toFixed(1)} GB / 枚)
          </span>
        </div>
        <input
          type="range"
          min="0.60"
          max="0.98"
          step="0.01"
          value={gpuMemoryUtilization}
          onChange={(e) => setGpuMemoryUtilization(parseFloat(e.target.value))}
          className="w-full accent-indigo-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-gray-500 dark:text-gray-400 font-mono mt-0.5">
          <span>60% (安全第一)</span>
          <span>90% (vLLM デフォルト)</span>
          <span>98% (極限利用)</span>
        </div>
      </div>
    </div>
  );
}
