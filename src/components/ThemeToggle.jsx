import React, { useState, useEffect, useCallback } from 'react';
import { Sun, Moon, Laptop } from 'lucide-react';

export default function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme-mode') || 'system';
  });

  const applyThemeToDom = useCallback((targetTheme) => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    let isDark = false;
    if (targetTheme === 'dark') {
      isDark = true;
    } else if (targetTheme === 'light') {
      isDark = false;
    } else {
      isDark = mediaQuery.matches;
    }

    if (isDark) {
      document.documentElement.classList.add('dark');
      document.body.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
      document.body.classList.remove('dark');
    }
  }, []);

  const handleSelectTheme = (newTheme) => {
    setTheme(newTheme);
    localStorage.setItem('theme-mode', newTheme);
    applyThemeToDom(newTheme);
  };

  useEffect(() => {
    applyThemeToDom(theme);

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = () => {
      const currentStored = localStorage.getItem('theme-mode') || 'system';
      if (currentStored === 'system') {
        applyThemeToDom('system');
      }
    };

    mediaQuery.addEventListener('change', handleSystemChange);
    return () => mediaQuery.removeEventListener('change', handleSystemChange);
  }, [theme, applyThemeToDom]);

  return (
    <div className="flex items-center p-0.5 bg-gray-200/90 dark:bg-gray-800/90 rounded-lg border border-gray-300 dark:border-gray-700 text-xs">
      <button
        type="button"
        onClick={() => handleSelectTheme('light')}
        title="ライトモード"
        className={`px-2.5 py-1 rounded-md flex items-center gap-1.5 transition ${
          theme === 'light'
            ? 'bg-white text-amber-600 shadow font-semibold'
            : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
        }`}
      >
        <Sun className="w-3.5 h-3.5 text-amber-500" />
        <span className="hidden sm:inline">ライト</span>
      </button>

      <button
        type="button"
        onClick={() => handleSelectTheme('dark')}
        title="ダークモード"
        className={`px-2.5 py-1 rounded-md flex items-center gap-1.5 transition ${
          theme === 'dark'
            ? 'bg-indigo-600 text-white shadow font-semibold'
            : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
        }`}
      >
        <Moon className="w-3.5 h-3.5" />
        <span className="hidden sm:inline">ダーク</span>
      </button>

      <button
        type="button"
        onClick={() => handleSelectTheme('system')}
        title="OS設定に連動 (自動)"
        className={`px-2.5 py-1 rounded-md flex items-center gap-1.5 transition ${
          theme === 'system'
            ? 'bg-indigo-600 text-white shadow font-semibold'
            : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
        }`}
      >
        <Laptop className="w-3.5 h-3.5" />
        <span className="hidden sm:inline">OS連動</span>
      </button>
    </div>
  );
}

