const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'SFX Smart Search',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    // Тёмная тема для frame
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#030712',
  });

  // В dev-режиме загружаем с dev-сервера
  const isDev = process.env.NODE_ENV === 'development';
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    // В production — из собранного dist/
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC: выбор директории с аудиофайлами
ipcMain.handle('select-audio-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Выберите папку с аудиофайлами',
  });
  return result.filePaths[0] || null;
});

// IPC: сканирование директории
ipcMain.handle('scan-directory', async (event, dirPath) => {
  const fs = require('fs');
  const path = require('path');
  
  const audioExtensions = ['.wav', '.mp3', '.ogg', '.flac', '.aiff'];
  const files = [];
  
  function scanDir(currentPath) {
    const entries = fs.readdirSync(currentPath, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentPath, entry.name);
      if (entry.isDirectory()) {
        scanDir(fullPath);
      } else if (audioExtensions.includes(path.extname(entry.name).toLowerCase())) {
        const stats = fs.statSync(fullPath);
        files.push({
          filename: entry.name,
          path: fullPath,
          size: stats.size,
          modified: stats.mtime.toISOString(),
        });
      }
    }
  }
  
  scanDir(dirPath);
  return files;
});
