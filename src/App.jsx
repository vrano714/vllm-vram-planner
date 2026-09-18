import React, { useState, useMemo } from 'react';
import { DEFAULT_MODEL_CONFIG } from './data/presets';
import { calculateVramAndRecommendations } from './utils/calculator';
import GpuConfig from './components/GpuConfig';
import ModelConfig from './components/ModelConfig';
import VllmSettings from './components/VllmSettings';
import VramVisualizer from './components/VramVisualizer';
import CommandOutput from './components/CommandOutput';
import ThemeToggle from './components/ThemeToggle';
import { RotateCcw, Layers } from 'lucide-react';

export default function App() {
  // 1. GPU State
  const [gpuVramGb, setGpuVramGb] = useState(24);
  const [gpuCount, setGpuCount] = useState(1);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState(0.90);

  // 2. Model State
  const [modelConfig, setModelConfig] = useState({
    modelId: DEFAULT_MODEL_CONFIG.id,
    paramsB: DEFAULT_MODEL_CONFIG.paramsB,
    layers: DEFAULT_MODEL_CONFIG.layers,
    hiddenSize: DEFAULT_MODEL_CONFIG.hiddenSize,
    attnHeads: DEFAULT_MODEL_CONFIG.attnHeads,
    kvHeads: DEFAULT_MODEL_CONFIG.kvHeads,
    maxModelLen: 8192,
    nativeMaxContext: DEFAULT_MODEL_CONFIG.maxContext,
    quantizationId: 'bfloat16',
    useDirectFileSize: false,
    customFileSizeGb: null,
    safetensorsSizeGb: null,
  });

  // 3. vLLM Serving & Tuning Settings
  const [vllmSettings, setVllmSettings] = useState({
    maxModelLen: 8192,
    kvCacheDtype: 'auto',
    enforceEager: false,
    enablePrefixCaching: true,
    enableChunkedPrefill: true,
    maxNumBatchedTokens: 2048,
    maxNumSeqs: 256,
    cpuOffloadGb: 0,
    port: 8000,
    host: '0.0.0.0',
    apiKey: '',
    servedModelName: '',
  });

  // Synchronize maxModelLen and detectedKvCacheDtype when model is loaded
  const handleModelLoaded = (updatedModel) => {
    setVllmSettings(prev => ({
      ...prev,
      maxModelLen: updatedModel.maxModelLen || prev.maxModelLen,
      kvCacheDtype: updatedModel.detectedKvCacheDtype || prev.kvCacheDtype,
    }));
  };

  // Combine state for calculator
  const fullConfig = useMemo(() => ({
    gpuVramGb,
    gpuCount,
    gpuMemoryUtilization,
    ...modelConfig,
    ...vllmSettings,
    maxModelLen: vllmSettings.maxModelLen,
  }), [gpuVramGb, gpuCount, gpuMemoryUtilization, modelConfig, vllmSettings]);

  // Run memory & concurrency calculation
  const calc = useMemo(() => {
    return calculateVramAndRecommendations(fullConfig);
  }, [fullConfig]);

  // Reset to default
  const handleReset = () => {
    setGpuVramGb(24);
    setGpuCount(1);
    setGpuMemoryUtilization(0.90);
    setModelConfig({
      modelId: DEFAULT_MODEL_CONFIG.id,
      paramsB: DEFAULT_MODEL_CONFIG.paramsB,
      layers: DEFAULT_MODEL_CONFIG.layers,
      hiddenSize: DEFAULT_MODEL_CONFIG.hiddenSize,
      attnHeads: DEFAULT_MODEL_CONFIG.attnHeads,
      kvHeads: DEFAULT_MODEL_CONFIG.kvHeads,
      maxModelLen: 8192,
      nativeMaxContext: DEFAULT_MODEL_CONFIG.maxContext,
      quantizationId: 'bfloat16',
      useDirectFileSize: false,
      customFileSizeGb: null,
      safetensorsSizeGb: null,
    });
    setVllmSettings({
      maxModelLen: 8192,
      kvCacheDtype: 'auto',
      enforceEager: false,
      enablePrefixCaching: true,
      enableChunkedPrefill: true,
      maxNumBatchedTokens: 2048,
      maxNumSeqs: 256,
      cpuOffloadGb: 0,
      port: 8000,
      host: '0.0.0.0',
      apiKey: '',
      servedModelName: '',
    });
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 dark:bg-[#0B0F19] dark:text-gray-100 flex flex-col transition-colors duration-200">
      {/* Top Navbar */}
      <header className="border-b border-gray-200 dark:border-gray-800/80 bg-white/90 dark:bg-[#0F1422]/90 backdrop-blur sticky top-0 z-50 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md shadow-indigo-500/20">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
                vLLM Configurator & VRAM Planner
                <span className="text-[10px] bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-700/60 px-2 py-0.5 rounded font-mono">
                  v1.0
                </span>
              </h1>
              <p className="text-xs text-amber-600 dark:text-amber-400/90 font-medium">
                ※モデル構造によっては計算が正しくない可能性があります
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <ThemeToggle />

            <button
              type="button"
              onClick={handleReset}
              className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
              title="設定を初期値に戻す"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>リセット</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Configuration Controls (7 cols) */}
          <div className="lg:col-span-6 space-y-6">
            <GpuConfig
              gpuVramGb={gpuVramGb}
              setGpuVramGb={setGpuVramGb}
              gpuCount={gpuCount}
              setGpuCount={setGpuCount}
              gpuMemoryUtilization={gpuMemoryUtilization}
              setGpuMemoryUtilization={setGpuMemoryUtilization}
            />

            <ModelConfig
              modelConfig={modelConfig}
              setModelConfig={setModelConfig}
              onModelLoaded={handleModelLoaded}
            />

            <VllmSettings
              settings={vllmSettings}
              setSettings={setVllmSettings}
            />
          </div>

          {/* Right Column: Visualizer & Generated Commands (6 cols) */}
          <div className="lg:col-span-6 space-y-6 lg:sticky lg:top-20">
            <VramVisualizer
              calc={calc}
              config={fullConfig}
            />

            <CommandOutput
              config={fullConfig}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800/80 py-4 bg-white dark:bg-[#0A0D16] text-center text-xs text-gray-500 dark:text-gray-400 transition-colors">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>vLLM Configurator & Memory Planner — Designed for LLM Serving Engineers</span>
          <span className="font-mono text-gray-600 dark:text-gray-400">Hugging Face API Direct Integration</span>
        </div>
      </footer>
    </div>
  );
}
