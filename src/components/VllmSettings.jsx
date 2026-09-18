import React, { useState } from 'react';
import { Settings2, HelpCircle, ChevronDown, ChevronUp } from 'lucide-react';

const CONTEXT_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

const formatContextLabel = (len) => {
  if (len >= 1048576) return `${len / 1048576}M`;
  if (len >= 1024) return `${len / 1024}k`;
  return len;
};

export default function VllmSettings({ settings, setSettings }) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const update = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm dark:shadow-lg space-y-4 transition-colors">
      <div className="flex items-center justify-between pb-3 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-lg">
          <Settings2 className="w-5 h-5" />
          <span>3. vLLM オプション & 最適化チューニング</span>
        </div>
      </div>

      {/* Context Window / max-model-len */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <label className="text-gray-700 dark:text-gray-300 font-medium flex items-center gap-1">
            <span>--max-model-len (コンテキスト最大トークン数)</span>
            <span className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 cursor-help" title="vLLMが処理する最大のシーケンス長。大きくするとKVキャッシュの消費が増加します。">
              <HelpCircle className="w-3.5 h-3.5" />
            </span>
          </label>
          <span className="font-mono text-indigo-600 dark:text-indigo-400 font-semibold">{settings.maxModelLen.toLocaleString()} tokens</span>
        </div>

        {/* Quick context pills */}
        <div className="flex flex-wrap gap-1.5">
          {CONTEXT_PRESETS.map((len) => (
            <button
              key={len}
              type="button"
              onClick={() => update('maxModelLen', len)}
              className={`px-2 py-1 rounded text-[11px] font-mono border transition ${
                settings.maxModelLen === len
                  ? 'bg-indigo-600 border-indigo-500 text-white font-medium'
                  : 'bg-gray-100 dark:bg-gray-800/60 border-gray-200 dark:border-gray-700/60 text-gray-700 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              {formatContextLabel(len)}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 pt-1">
          <input
            type="range"
            min="1024"
            max="1048576"
            step="1024"
            value={settings.maxModelLen}
            onChange={(e) => update('maxModelLen', parseInt(e.target.value, 10))}
            className="flex-1 accent-indigo-500 cursor-pointer"
          />
          <input
            type="number"
            min="512"
            max="2097152"
            step="1024"
            value={settings.maxModelLen}
            onChange={(e) => update('maxModelLen', Math.max(1, parseInt(e.target.value, 10) || 1024))}
            className="w-24 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 text-center font-mono text-xs text-gray-900 dark:text-white focus:outline-none focus:border-indigo-500"
          />
        </div>
      </div>

      {/* KV Cache Dtype */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80">
        <div className="flex justify-between items-center text-xs mb-1.5">
          <label className="text-gray-700 dark:text-gray-300 font-medium flex items-center gap-1">
            <span>--kv-cache-dtype (KVキャッシュの精度)</span>
          </label>
          <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-mono font-medium">
            {settings.kvCacheDtype === 'fp8' ? '★ メモリ消費 50% 削減' : '標準精度 (BF16/FP16)'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => update('kvCacheDtype', 'auto')}
            className={`p-2.5 rounded-lg text-xs font-medium border text-left transition ${
              settings.kvCacheDtype === 'auto'
                ? 'bg-indigo-50 dark:bg-indigo-950/40 border-indigo-500 text-indigo-900 dark:text-white ring-1 ring-indigo-500/50'
                : 'bg-gray-50 dark:bg-gray-800/40 border-gray-200 dark:border-gray-700/60 text-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            <div className="font-semibold text-gray-900 dark:text-white">auto (通常)</div>
            <div className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">2 Bytes / element (高精度)</div>
          </button>
          <button
            type="button"
            onClick={() => update('kvCacheDtype', 'fp8')}
            className={`p-2.5 rounded-lg text-xs font-medium border text-left transition ${
              settings.kvCacheDtype === 'fp8'
                ? 'bg-emerald-50 dark:bg-indigo-950/40 border-emerald-500 dark:border-indigo-500 text-emerald-900 dark:text-white ring-1 ring-emerald-500/50 dark:ring-indigo-500/50'
                : 'bg-gray-50 dark:bg-gray-800/40 border-gray-200 dark:border-gray-700/60 text-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-200'
            }`}
          >
            <div className="font-semibold text-emerald-600 dark:text-emerald-400">fp8 (VRAM大幅節約)</div>
            <div className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">1 Byte / element (同時処理数2倍)</div>
          </button>
        </div>
      </div>

      {/* Critical Performance & Memory Switches */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80 space-y-2.5">
        <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">重要フラグ & メモリ制御</label>

        {/* enforce-eager */}
        <label className="flex items-start gap-3 p-2.5 bg-gray-50 dark:bg-gray-900/60 hover:bg-gray-100 dark:hover:bg-gray-900 border border-gray-200 dark:border-gray-800/80 rounded-lg cursor-pointer transition">
          <input
            type="checkbox"
            checked={settings.enforceEager}
            onChange={(e) => update('enforceEager', e.target.checked)}
            className="mt-0.5 w-4 h-4 rounded text-indigo-600 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 focus:ring-0"
          />
          <div className="text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-gray-900 dark:text-white">
              <span>--enforce-eager (CUDA Graph キャプチャ無効化)</span>
              <span className="bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-800/80 text-[10px] px-1.5 py-0.2 rounded font-mono">
                VRAM 約1.2GB 節約
              </span>
            </div>
            <p className="text-gray-500 dark:text-gray-400 text-[11px] mt-0.5">
              PyTorch Eagerモードで実行。CUDA Graphのメモリ確保をスキップし、空き容量ギリギリの環境でのOOMを防ぎます。
            </p>
          </div>
        </label>

        {/* enable-prefix-caching */}
        <label className="flex items-start gap-3 p-2.5 bg-gray-50 dark:bg-gray-900/60 hover:bg-gray-100 dark:hover:bg-gray-900 border border-gray-200 dark:border-gray-800/80 rounded-lg cursor-pointer transition">
          <input
            type="checkbox"
            checked={settings.enablePrefixCaching}
            onChange={(e) => update('enablePrefixCaching', e.target.checked)}
            className="mt-0.5 w-4 h-4 rounded text-indigo-600 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 focus:ring-0"
          />
          <div className="text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-gray-900 dark:text-white">
              <span>--enable-prefix-caching (プレフィックスキャッシュ)</span>
              <span className="bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800/80 text-[10px] px-1.5 py-0.2 rounded font-mono">
                RAG / 会話高速化
              </span>
            </div>
            <p className="text-gray-500 dark:text-gray-400 text-[11px] mt-0.5">
              共通のシステムプロンプトやマルチターンチャット履歴のKVキャッシュを共有・再利用し、TTFT（最初のトークン出力速度）を劇的に向上させます。
            </p>
          </div>
        </label>

        {/* enable-chunked-prefill */}
        <label className="flex items-start gap-3 p-2.5 bg-gray-50 dark:bg-gray-900/60 hover:bg-gray-100 dark:hover:bg-gray-900 border border-gray-200 dark:border-gray-800/80 rounded-lg cursor-pointer transition">
          <input
            type="checkbox"
            checked={settings.enableChunkedPrefill}
            onChange={(e) => update('enableChunkedPrefill', e.target.checked)}
            className="mt-0.5 w-4 h-4 rounded text-indigo-600 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-700 focus:ring-0"
          />
          <div className="text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-gray-900 dark:text-white">
              <span>--enable-chunked-prefill (Prefillチャンク分割)</span>
              <span className="bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-800/80 text-[10px] px-1.5 py-0.2 rounded font-mono">
                レイテンシ安定化
              </span>
            </div>
            <p className="text-gray-500 dark:text-gray-400 text-[11px] mt-0.5">
              長文リクエスト処理中に新しいリクエストが入っても、トークン生成が詰まるのを防ぎスループットを平滑化します。
            </p>
          </div>
        </label>
      </div>

      {/* Advanced / Serving Settings Toggle */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800/80">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 flex items-center justify-between w-full py-1 transition"
        >
          <span>サービング・バッチサイズ・ネットワーク設定</span>
          {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 gap-3 mt-2 bg-gray-50 dark:bg-gray-900/60 p-3 rounded-lg border border-gray-200 dark:border-gray-800 text-xs">
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">--port (ポート番号)</label>
              <input
                type="number"
                value={settings.port}
                onChange={(e) => update('port', parseInt(e.target.value, 10) || 8000)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">--host (バインドアドレス)</label>
              <input
                type="text"
                value={settings.host}
                onChange={(e) => update('host', e.target.value)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">--served-model-name (公開モデル別名)</label>
              <input
                type="text"
                placeholder="例: gpt-4o, my-model"
                value={settings.servedModelName}
                onChange={(e) => update('servedModelName', e.target.value)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">--api-key (API認証キー)</label>
              <input
                type="password"
                placeholder="未設定時は認証なし"
                value={settings.apiKey}
                onChange={(e) => update('apiKey', e.target.value)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">--max-num-seqs (最大同時シーケンス数)</label>
              <input
                type="number"
                value={settings.maxNumSeqs}
                onChange={(e) => update('maxNumSeqs', parseInt(e.target.value, 10) || 256)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="text-gray-600 dark:text-gray-400 block mb-1">--cpu-offload-gb (CPUオフロード GB)</label>
              <input
                type="number"
                min="0"
                max="128"
                value={settings.cpuOffloadGb}
                onChange={(e) => update('cpuOffloadGb', parseInt(e.target.value, 10) || 0)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded px-2.5 py-1 text-gray-900 dark:text-white font-mono"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
