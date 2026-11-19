// Preload bridge for electron/main.js (was missing from the project).
// Exposes folder-pick + directory-scan IPC to the renderer.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('sfxApi', {
  selectAudioDirectory: () => ipcRenderer.invoke('select-audio-directory'),
  scanDirectory: (dirPath) => ipcRenderer.invoke('scan-directory', dirPath),
});
