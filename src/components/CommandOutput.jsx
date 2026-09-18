import React, { useState } from 'react';
import { Terminal, Copy, Check, FileCode, Container } from 'lucide-react';
import { generateCommands } from '../utils/calculator';

export default function CommandOutput({ config }) {
  const [activeTab, setActiveTab] = useState('cli');
  const [copied, setCopied] = useState(false);
  const [singleLine, setSingleLine] = useState(false);

  const commands = generateCommands(config);

  let currentContent = '';
  if (activeTab === 'cli') {
    currentContent = singleLine ? commands.cliSingleLine : commands.cliMultiLine;
  } else if (activeTab === 'docker') {
    currentContent = commands.dockerCmd;
  } else if (activeTab === 'compose') {
    currentContent = commands.composeYaml;
  } else if (activeTab === 'python') {
    currentContent = commands.pythonCode;
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy', err);
    }
  };

  return (
    <div className="bg-white dark:bg-[#111827] border border-gray-200 dark:border-gray-800 rounded-xl p-5 shadow-sm dark:shadow-lg space-y-3 transition-colors">
      {/* Header & Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold text-lg">
          <Terminal className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          <span>5. 生成されたサービングコマンド</span>
        </div>

        {/* Tab Buttons */}
        <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-900/90 p-1 rounded-lg border border-gray-200 dark:border-gray-800 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('cli')}
            className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
              activeTab === 'cli'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>CLI (vllm serve)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('docker')}
            className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
              activeTab === 'docker'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Container className="w-3.5 h-3.5" />
            <span>Docker Run</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('compose')}
            className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
              activeTab === 'compose'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            <span>Compose YAML</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('python')}
            className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
              activeTab === 'python'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            <span>Python</span>
          </button>
        </div>
      </div>

      {/* Code Viewer Container */}
      <div className="relative rounded-lg overflow-hidden border border-gray-800 bg-[#090D16]">
        {/* Actions bar */}
        <div className="flex justify-between items-center px-4 py-2 bg-gray-900/80 border-b border-gray-800/80 text-xs">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80"></span>
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80"></span>
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80"></span>
            </div>
            <span className="text-[11px] font-mono text-gray-500 ml-2">
              {activeTab === 'cli' ? 'bash' : activeTab === 'docker' ? 'bash' : activeTab === 'compose' ? 'yaml' : 'python'}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {activeTab === 'cli' && (
              <label className="flex items-center gap-1.5 text-[11px] text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={singleLine}
                  onChange={(e) => setSingleLine(e.target.checked)}
                  className="rounded text-indigo-600 bg-gray-800 border-gray-700"
                />
                <span>1行で表示</span>
              </label>
            )}

            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1.5 bg-gray-800 hover:bg-gray-700 text-gray-200 px-3 py-1 rounded text-xs font-medium transition active:scale-95 border border-gray-700"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">コピー完了！</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>コマンドをコピー</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Code Block */}
        <pre className="p-4 text-xs font-mono text-gray-200 overflow-x-auto selection:bg-indigo-500 selection:text-white leading-relaxed">
          <code>{currentContent}</code>
        </pre>
      </div>

    </div>
  );
}
