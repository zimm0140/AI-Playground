import {
  app,
  BrowserWindow,
  shell,
  ipcMain,
  screen,
  IpcMainEvent,
  IpcMainInvokeEvent,
  dialog,
  OpenDialogSyncOptions,
  MessageBoxSyncOptions,
  MessageBoxOptions,
} from "electron";
import path from "node:path";
import fs from "fs";
import { exec, spawn, ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import koffi from 'koffi';
import sudo from "sudo-prompt";
import { PathsManager } from "./pathsManager";
import { createLogger, format, transports } from 'winston';

// Set up the logger
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.printf(({ timestamp, level, message }) => `${timestamp} [${level.toUpperCase()}]: ${message}`)
  ),
  transports: [
    new transports.Console(),
    new transports.File({ filename: 'error.log', level: 'error' }),
    new transports.File({ filename: 'combined.log' }),
  ],
});

// Setting up environment variables
process.env.DIST = path.join(__dirname, "../");
process.env.VITE_PUBLIC = path.join(__dirname, app.isPackaged ? "../.." : "../../../public");

const externalRes = path.resolve(app.isPackaged
  ? process.resourcesPath
  : path.join(__dirname, "../../external/"));

const singleLock = app.requestSingleInstanceLock();
let win: BrowserWindow | null;
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
const appSize = { width: 820, height: 128, maxChatContentHeight: 0 };
const settings: LocalSettings = {
  apiHost: "http://127.0.0.1:9999",
  settingPath: "",
  isAdminExec: false,
  debug: 0,
  envType: "ultra",
  port: 59999,
  windowsShell: 'powershell.exe',
};

function loadSettings() {
  const settingPath = app.isPackaged
    ? path.join(process.resourcesPath, "settings.json")
    : path.join(__dirname, "../../external/settings-dev.json");

  if (fs.existsSync(settingPath)) {
    const loadedSettings = JSON.parse(fs.readFileSync(settingPath, { encoding: "utf8" }));
    Object.keys(loadedSettings).forEach((key) => {
      if (key in settings) settings[key] = loadedSettings[key];
    });
  }
  settings.apiHost = `http://127.0.0.1:${settings.port}`;
}

async function createWindow() {
  win = new BrowserWindow({
    title: "AI PLAYGROUND",
    icon: path.join(process.env.VITE_PUBLIC, "app-ico.svg"),
    transparent: true,
    resizable: false,
    frame: false,
    width: 1440,
    height: 951,
    webPreferences: {
      preload: path.join(__dirname, "../preload/preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      enableRemoteModule: false,
      sandbox: true
    },
  });

  const session = win.webContents.session;
  if (!app.isPackaged || settings.debug) {
    win.webContents.openDevTools({ mode: "detach", activate: true });
  }

  session.webRequest.onBeforeSendHeaders((details, callback) => {
    callback({ requestHeaders: { ...details.requestHeaders, Origin: "*" } });
  });

  session.webRequest.onHeadersReceived((details, callback) => {
    if (details.url.startsWith(settings.apiHost)) {
      details.responseHeaders = {
        ...details.responseHeaders,
        "Access-Control-Allow-Origin": ["*"],
        "Access-Control-Allow-Methods": ["GET,POST"],
        "Access-Control-Allow-Headers": ["x-requested-with,Content-Type"],
      };
    }
    callback(details);
  });

  win.webContents.session.setPermissionRequestHandler((_, permission, callback) => {
    const allowedPermissions = ["media", "clipboard-sanitized-write"];
    callback(allowedPermissions.includes(permission));
  });

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
    logger.info(`Loaded URL: ${VITE_DEV_SERVER_URL}`);
  } else {
    win.loadFile(path.join(process.env.DIST, "index.html"));
  }

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https:")) shell.openExternal(url);
    return { action: "deny" };
  });
}

function logMessage(message: string) {
  if (app.isPackaged) {
    logger.info(message);
    fs.appendFileSync(path.join(externalRes, "debug.log"), message + "\r\n");
  } else {
    logger.info(message);
  }
}

app.on("quit", async () => {
  if (singleLock) app.releaseSingleInstanceLock();
});

app.on("window-all-closed", async () => {
  try {
    await closeApiService();
  } catch (err) {
    logger.error('Error during closeApiService:', err);
  }
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

function initEventHandle() {
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    if (win && !win.isDestroyed()) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  screen.on("display-metrics-changed", (event, display, changedMetrics) => {
    if (win) {
      win.setBounds({
        x: 0,
        y: 0,
        width: display.workAreaSize.width,
        height: display.workAreaSize.height,
      });
      win.webContents.send("display-metrics-changed", display.workAreaSize.width, display.workAreaSize.height);
    }
  });

  ipcMain.handle("getLocalSettings", async () => ({
    apiHost: settings.apiHost,
    showIndex: settings.showIndex,
    showBenchmark: settings.showBenchmark,
    isAdminExec: isAdmin(),
  }));

  ipcMain.handle("getWinSize", () => appSize);

  ipcMain.on("openUrl", (event, url: string) => shell.openExternal(url));

  ipcMain.handle("setWinSize", (event: IpcMainInvokeEvent, width: number, height: number) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    const winRect = win.getBounds();
    if (winRect.width !== width || winRect.height !== height) {
      const y = winRect.y + (winRect.height - height);
      win.setBounds({ x: winRect.x, y, width, height });
    }
  });

  ipcMain.handle("restorePathsSettings", (event: IpcMainInvokeEvent) => {
    const paths = app.isPackaged ? {
      "llm": "./resources/service/models/llm/checkpoints",
      "embedding": "./resources/service/models/llm/embedding",
      "stableDiffusion": "./resources/service/models/stable_diffusion/checkpoints",
      "inpaint": "./resources/service/models/stable_diffusion/inpaint",
      "lora": "./resources/service/models/stable_diffusion/lora",
      "vae": "./resources/service/models/stable_diffusion/vae"
    } : {
      "llm": "../service/models/llm/checkpoints",
      "embedding": "../service/models/llm/embedding",
      "stableDiffusion": "../service/models/stable_diffusion/checkpoints",
      "inpaint": "../service/models/stable_diffusion/inpaint",
      "lora": "../service/models/stable_diffusion/lora",
      "vae": "../service/models/stable_diffusion/vae"
    };
    pathsManager.updateModelPaths(paths);
  });

  ipcMain.on("miniWindow", () => win && win.minimize());
  ipcMain.on("setFullScreen", (event: IpcMainEvent, enable: boolean) => win && win.setFullScreen(enable));
  ipcMain.on("exitApp", async () => win && win.close());

  ipcMain.on("saveImage", async (event: IpcMainEvent, url: string) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) return;
    const options = {
      title: "Save Image",
      defaultPath: path.join(app.getPath("documents"), "example.png"),
      filters: [{ name: "AIGC-Generate.png", extensions: ["png"] }],
    };
    try {
      const result = await dialog.showSaveDialog(win, options);
      if (!result.canceled && result.filePath) {
        if (fs.existsSync(result.filePath)) fs.rmSync(result.filePath);
        try {
          const response = await fetch(url);
          const arrayBuffer = await response.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          fs.writeFileSync(result.filePath, buffer);
          logger.info(`File downloaded and saved: ${result.filePath}`);
        } catch (error) {
          logger.error("Download and save error:", error);
        }
      }
    } catch (err) {
      logger.error("Error during saveImage:", err);
    }
  });

  ipcMain.handle("showOpenDialog", async (event, options: OpenDialogSyncOptions) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    return await dialog.showOpenDialog(win, options);
  });

  ipcMain.handle("showMessageBox", async (event, options: MessageBoxOptions) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    return dialog.showMessageBox(win, options);
  });

  ipcMain.handle("showMessageBoxSync", async (event, options: MessageBoxSyncOptions) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    return dialog.showMessageBoxSync(win, options);
  });

  ipcMain.handle("existsPath", async (event, path: string) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    return win ? fs.existsSync(path) : false;
  });

  let pathsManager = new PathsManager(path.join(externalRes, app.isPackaged ? "model_config.json" : "model_config.dev.json"));

  ipcMain.handle("getInitSetting", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) return;
    return {
      apiHost: settings.apiHost,
      modelLists: pathsManager.scanAll(),
      modelPaths: pathsManager.modelPaths,
      envType: settings.envType,
      isAdminExec: settings.isAdminExec,
      version: app.getVersion()
    };
  });

  ipcMain.handle("updateModelPaths", (event, modelPaths: ModelPaths) => {
    pathsManager.updateModelPaths(modelPaths);
    return pathsManager.scanAll();
  });

  ipcMain.handle("refreshSDModels", (event) => pathsManager.scanSDModelLists());
  ipcMain.handle("refreshInpaintModels", (event) => pathsManager.scanInpaint());
  ipcMain.handle("refreshLora", (event) => pathsManager.scanLora());
  ipcMain.handle("refreshLLMModels", (event) => pathsManager.scanLLMModels());

  ipcMain.on("openImageWithSystem", (event, url: string) => {
    let imagePath = url.replace(settings.apiHost + "/", ""); 
    imagePath = app.isPackaged 
      ? path.join(externalRes, "service", imagePath) 
      : path.join(app.getAppPath(), "../service", imagePath);
    shell.openPath(imagePath);
  });

  ipcMain.on("selectImage", (event, url: string) => {
    let imagePath = url.replace(settings.apiHost + "/", ""); 
    imagePath = app.isPackaged 
      ? path.join(externalRes, "service", imagePath) 
      : path.join("..", "service", imagePath);

    if (process.platform === 'win32') {
      exec(`explorer.exe /select, "${imagePath}"`);
    } else {
      shell.showItemInFolder(imagePath);
    }
  });
}

const apiService: {
  webProcess: ChildProcess | null,
  normalExit: boolean
} = { webProcess: null, normalExit: true };

function isProcessRunning(pid: number) {
  try {
    return process.kill(pid, 0);
  } catch {
    return false;
  }
}

function wakeupApiService() {
  const wordkDir = path.resolve(app.isPackaged ? path.join(process.resourcesPath, "service") : path.join(__dirname, "../../../service"));
  const baseDir = app.isPackaged ? process.resourcesPath : path.join(__dirname, "../../../");

  const pythonExe = process.platform === 'win32'
    ? path.resolve(path.join(baseDir, "env/python.exe"))
    : path.resolve(path.join(baseDir, "env/bin/python3"));

  const newEnv = {
    "SYCL_ENABLE_DEFAULT_CONTEXTS": "1",
    "SYCL_CACHE_PERSISTENT": "1",
    "PYTHONIOENCODING": "utf-8"
  };

  const mergedEnv = { ...process.env, ...newEnv };

  const options = {
    cwd: wordkDir,
    detached: settings.debug, // Detached in debug mode
    windowsHide: !settings.debug, // Hide window in production
    env: mergedEnv
  };

  const isWindows = process.platform === 'win32';
  const shell = isWindows 
    ? (settings.debug ? 'cmd.exe' : settings.windowsShell) // Use the specified shell
    : '/bin/sh'; 
  const shellArgs = isWindows ? ['/c', pythonExe, 'web_api.py'] : ['-c', `${pythonExe} web_api.py`];

  try {
    apiService.webProcess = spawn(shell, shellArgs, options);
    apiService.webProcess.on('error', (err) => logger.error('Failed to start subprocess:', err));
  } catch (err) {
    logger.error('Error spawning process:', err);
  }
}

function closeApiService() {
  apiService.normalExit = true;
  if (apiService.webProcess && apiService.webProcess.pid && isProcessRunning(apiService.webProcess.pid)) {
    apiService.webProcess.kill();
    apiService.webProcess = null;
  }
  return fetch(`${settings.apiHost}/api/applicationExit`)
    .then(response => {
      if (!response.ok) {
        logger.error(`API shutdown failed: ${response.status} - ${response.statusText}`);
      } else {
        logger.info("API service shutdown successfully."); 
      }
    })
    .catch(error => {
      logger.error('Error during API exit:', error); 
    });
}

ipcMain.on("openImageWin", (_: IpcMainEvent, url: string, title: string, width: number, height: number) => {
  const display = screen.getPrimaryDisplay();
  width += 32;
  height += 48;
  width = Math.min(width, display.workAreaSize.width);
  height = Math.min(height, display.workAreaSize.height);

  const imgWin = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, "app-ico.svg"),
    resizable: true,
    center: true,
    frame: true,
    width,
    height,
    autoHideMenuBar: true,
    show: false,
    parent: win || undefined,
    webPreferences: {
      devTools: false
    }
  });
  imgWin.setMenu(null);
  imgWin.loadURL(url);
  imgWin.once("ready-to-show", () => {
    imgWin.show();
    imgWin.setTitle(title);
  });
});

ipcMain.handle('showSaveDialog', async (event, options: Electron.SaveDialogOptions) => {
  return dialog.showSaveDialog(options).catch(err => logger.error('Error in showSaveDialog:', err));
});

function needAdminPermission(): Promise<boolean> {
  return new Promise<boolean>((resolve, reject) => { // Use reject for errors
    if (process.platform === 'win32') {
      const filename = path.join(externalRes, `${randomUUID()}.txt`);
      fs.writeFile(filename, '', (err) => {
        if (err) {
          if (err.code === 'EPERM' && path.parse(externalRes).root === path.parse(process.env.windir!).root) {
            resolve(true);
          } else {
            logger.error('Failed to write file for admin check:', err);
            resolve(false);
          }
        } else {
          fs.rmSync(filename);
          resolve(false);
        }
      });
    } else {
      fs.access(externalRes, fs.constants.W_OK, (err) => {
        if (err) {
          logger.error('Failed to access external resource directory:', err);
          reject(err); // Handle errors
        } else {
          resolve(false);
        }
      });
    }
  });
}

function isAdmin(): boolean {
  if (process.platform === 'win32') {
    const lib = koffi.load("Shell32.dll");
    try {
      const IsUserAnAdmin = lib.func("IsUserAnAdmin", "bool", []);
      return IsUserAnAdmin();
    } finally {
      lib.unload();
    }
  } else {
    return process.getuid() === 0;
  }
}

app.whenReady().then(async () => {
  try {
    if (await needAdminPermission()) {
      if (singleLock) app.releaseSingleInstanceLock();
      sudo.exec(process.argv.join(' ').trim(), (err, stdout, stderr) => {
        if (err) logger.error("Sudo exec error:", err);
        app.exit(0);
      });
      return;
    }

    if (!singleLock) {
      dialog.showMessageBoxSync({
        message: app.getLocale() === "zh-CN" ? "本程序仅允许单实例运行，确认后本次运行将自动结束" : "This program only allows a single instance to run, and the run will automatically end after confirmation",
        title: "error",
        type: "error"
      });
      app.exit();
    } else {
      loadSettings();
      initEventHandle();
      createWindow();
      wakeupApiService();
    }
  } catch (error) {
    logger.error("Error during app initialization:", error);
  }
});
