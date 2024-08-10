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

/**
 * @section Application Logging Setup
 * 
 * Initialize the Winston logger for structured logging to the console and files.
 * This provides detailed logs with timestamps and log levels for easier debugging and monitoring. 
 */
const logger = createLogger({
  level: 'info', // Set the minimum log level to 'info'
  format: format.combine(
    format.timestamp(), // Include a timestamp in each log entry
    format.printf(({ timestamp, level, message }) => `${timestamp} [${level.toUpperCase()}]: ${message}`) // Define the log output format
  ),
  transports: [
    new transports.Console(), // Log to the console 
    new transports.File({ filename: 'error.log', level: 'error' }), // Log errors to a dedicated file
    new transports.File({ filename: 'combined.log' }), // Log all messages to a combined file
  ],
});

/**
 * @section Application Environment and Configuration
 * 
 * Set up environment variables and application constants, including paths, settings, 
 * and ensure single-instance application execution.
 */

// Set environment variables for application paths.
process.env.DIST = path.join(__dirname, "../"); // Path to the distributable directory.
process.env.VITE_PUBLIC = path.join(__dirname, app.isPackaged ? "../.." : "../../../public"); // Path to public assets.

// Resolve the path to the external resources directory based on whether the application is packaged.
const externalRes = path.resolve(app.isPackaged
  ? process.resourcesPath
  : path.join(__dirname, "../../external/"));

// Ensure that only one instance of the application is running.
const singleLock = app.requestSingleInstanceLock(); 

// Global reference to the main application window. 
let win: BrowserWindow | null; 

// URL of the Vite development server, if the application is running in development mode.
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"]; 

// Define default application window size.
const appSize = { width: 820, height: 128, maxChatContentHeight: 0 }; 

/**
 * Application settings that control various aspects of the application's behavior.
 * These settings can be overridden by loading a settings file. 
 */
const settings: LocalSettings = {
  apiHost: "http://127.0.0.1:9999", // Default API server address.
  settingPath: "", // Path to the settings file.
  isAdminExec: false, // Flag indicating if the application is running with administrator privileges.
  debug: 0, // Debug mode flag. Set to 1 to enable debug mode.
  envType: "ultra", // Environment type setting.
  port: 59999, // Port for the API service.
  windowsShell: 'powershell.exe', // Preferred shell to use on Windows in non-debug mode.
};

/**
 * @section Application Initialization
 * 
 * Functions for loading settings, creating the main window, 
 * and handling core application events.
 */

/**
 * Load application settings from a JSON file.
 * Merges loaded settings with the default settings.
 */
function loadSettings() {
  // Determine the settings file path based on whether the application is packaged.
  const settingPath = app.isPackaged
    ? path.join(process.resourcesPath, "settings.json") // Path for packaged app
    : path.join(__dirname, "../../external/settings-dev.json"); // Path for development

  if (fs.existsSync(settingPath)) {
    const loadedSettings = JSON.parse(fs.readFileSync(settingPath, { encoding: "utf8" }));

    // Update the application settings with the loaded settings. 
    Object.keys(loadedSettings).forEach((key) => {
      if (key in settings) {
        settings[key] = loadedSettings[key];
      }
    });
  }
  settings.apiHost = `http://127.0.0.1:${settings.port}`; // Update the API host based on the loaded port
}

/**
 * Create the main application window.
 * This function sets up the BrowserWindow, handles security configurations, 
 * loads the application content, and manages external links.
 */
async function createWindow() {
  // Create the browser window
  win = new BrowserWindow({
    title: "AI PLAYGROUND",
    icon: path.join(process.env.VITE_PUBLIC, "app-ico.svg"),
    transparent: true, // Make the window transparent
    resizable: false, // Prevent the user from resizing the window
    frame: false, // Remove the default window frame
    width: 1440, // Set the initial window width
    height: 951, // Set the initial window height
    webPreferences: {
      preload: path.join(__dirname, "../preload/preload.js"), // Load the preload script
      contextIsolation: true, // Enable context isolation for security 
      nodeIntegration: false, // Disable Node.js integration in the renderer process for security
      enableRemoteModule: false, // Disable the remote module for security
      sandbox: true // Enable sandboxing for enhanced security
    },
  });

  const session = win.webContents.session; // Get the webContents session for the window

  // Open DevTools if not packaged or in debug mode
  if (!app.isPackaged || settings.debug) {
    win.webContents.openDevTools({ mode: "detach", activate: true });
  }

  // Set CORS headers for all outgoing requests
  session.webRequest.onBeforeSendHeaders((details, callback) => {
    callback({ requestHeaders: { ...details.requestHeaders, Origin: "*" } });
  });

  // Adjust response headers to handle CORS for requests made to the API host
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

  // Define allowed permissions for the application window
  win.webContents.session.setPermissionRequestHandler((_, permission, callback) => {
    const allowedPermissions = ["media", "clipboard-sanitized-write"];
    callback(allowedPermissions.includes(permission)); // Only allow the listed permissions
  });

  // Load the application content based on the environment
  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL); // Load from the development server 
    logger.info(`Loaded URL: ${VITE_DEV_SERVER_URL}`); 
  } else {
    win.loadFile(path.join(process.env.DIST, "index.html")); // Load from the packaged index.html file 
  }

  // Ensure that external links open in the user's default browser
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https:")) shell.openExternal(url);
    return { action: "deny" }; // Prevent opening links within the Electron app
  });
}

/**
 * Log a message to the console and to a file if the application is packaged.
 * @param {string} message The message to be logged.
 */
function logMessage(message: string) {
  logger.info(message); // Log to console always
  if (app.isPackaged) {
    fs.appendFileSync(path.join(externalRes, "debug.log"), message + "\r\n"); // Append to file if packaged
  } 
}

/**
 * @section Application Lifecycle Events
 * 
 * Event handlers for managing the application lifecycle, 
 * including quitting, closing windows, activating the app, 
 * and handling second instances. 
 */

// Release the single instance lock when the app quits.
app.on("quit", async () => {
  if (singleLock) app.releaseSingleInstanceLock(); 
});

// Quit when all windows are closed (except on macOS).
// Attempts to gracefully shut down the API service before quitting.
app.on("window-all-closed", async () => {
  try {
    await closeApiService();
  } catch (err) {
    logger.error(`Error during closeApiService: ${err}`);
  }
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});

// On macOS, re-create the window when the dock icon is clicked 
// and there are no other windows open.
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

/**
 * @section Inter-Process Communication (IPC) and Event Handling
 *
 * Initialize various event handlers for handling inter-process communication (IPC) events,
 * managing window state, responding to screen events, and interacting with the file system.
 */
function initEventHandle() {

  // Restore and focus the main window if a second instance of the app is launched.
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    if (win && !win.isDestroyed()) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  // Adjust the main window bounds when display metrics change. 
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

  // Handle IPC events 
  // ----------------------------------

  // Return the current local settings to the renderer process. 
  ipcMain.handle("getLocalSettings", async () => ({
    apiHost: settings.apiHost,
    showIndex: settings.showIndex,
    showBenchmark: settings.showBenchmark,
    isAdminExec: isAdmin(),
  }));

  // Return the default window size to the renderer process.
  ipcMain.handle("getWinSize", () => appSize);

  // Open a URL in the default system browser.
  ipcMain.on("openUrl", (event, url: string) => shell.openExternal(url));

  // Set the application window size based on a request from the renderer process.
  ipcMain.handle("setWinSize", (event: IpcMainInvokeEvent, width: number, height: number) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    const winRect = win.getBounds();
    if (winRect.width !== width || winRect.height !== height) {
      const y = winRect.y + (winRect.height - height);
      win.setBounds({ x: winRect.x, y, width, height });
    }
  });

  // Restore default paths for model files based on whether the application is packaged.
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

  // Minimize the application window. 
  ipcMain.on("miniWindow", () => win && win.minimize());

  // Set the application window to fullscreen mode based on a request from the renderer. 
  ipcMain.on("setFullScreen", (event: IpcMainEvent, enable: boolean) => win && win.setFullScreen(enable));

  // Exit the application.
  ipcMain.on("exitApp", async () => win && win.close());

  // Save an image to the user's file system.
  ipcMain.on("saveImage", async (event: IpcMainEvent, url: string) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) return;

    // Set up options for the save dialog.
    const options = {
      title: "Save Image",
      defaultPath: path.join(app.getPath("documents"), "example.png"),
      filters: [{ name: "AIGC-Generate.png", extensions: ["png"] }],
    };

    try {
      const result = await dialog.showSaveDialog(win, options);
      if (!result.canceled && result.filePath) {
        if (fs.existsSync(result.filePath)) {
          fs.rmSync(result.filePath); // Delete the file if it already exists
        }
        try {
          const response = await fetch(url); // Fetch the image from the URL
          const arrayBuffer = await response.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          fs.writeFileSync(result.filePath, buffer); // Save the image to the selected file path
          logger.info(`File downloaded and saved: ${result.filePath}`);
        } catch (error) {
          logger.error("Download and save error:", error);
        }
      }
    } catch (err) {
      logger.error("Error during saveImage:", err);
    }
  });

  // Show an open file dialog. 
  ipcMain.handle("showOpenDialog", async (event, options: OpenDialogSyncOptions) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    return await dialog.showOpenDialog(win, options);
  });

  // Show a message box to the user.
  ipcMain.handle("showMessageBox", async (event, options: MessageBoxOptions) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    return dialog.showMessageBox(win, options);
  });

  // Show a synchronous message box to the user. 
  ipcMain.handle("showMessageBoxSync", async (event, options: MessageBoxSyncOptions) => {
    const win = BrowserWindow.fromWebContents(event.sender)!;
    return dialog.showMessageBoxSync(win, options);
  });

  // Check if a file path exists.
  ipcMain.handle("existsPath", async (event, path: string) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    return win ? fs.existsSync(path) : false;
  });

  // Initialize the PathsManager for managing model configuration file paths.
  let pathsManager = new PathsManager(path.join(externalRes, app.isPackaged ? "model_config.json" : "model_config.dev.json"));

  // Get the initial settings for the application.
  ipcMain.handle("getInitSetting", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) return;
    return {
      apiHost: settings.apiHost,
      modelLists: pathsManager.scanAll(), // Get a list of all available models
      modelPaths: pathsManager.modelPaths, // Get the configured model paths
      envType: settings.envType,
      isAdminExec: settings.isAdminExec,
      version: app.getVersion() // Get the application version
    };
  });

  // Handle requests to update model paths. 
  ipcMain.handle("updateModelPaths", (event, modelPaths: ModelPaths) => {
    pathsManager.updateModelPaths(modelPaths); // Update the model paths
    return pathsManager.scanAll(); // Return the updated list of models
  });

  // Handle IPC events for refreshing model lists 
  // -----------------------------------------------

  // Refresh Stable Diffusion models.
  ipcMain.handle("refreshSDModels", (event) => pathsManager.scanSDModelLists());
  // Refresh Inpaint models.
  ipcMain.handle("refreshInpaintModels", (event) => pathsManager.scanInpaint());
  // Refresh Lora models.
  ipcMain.handle("refreshLora", (event) => pathsManager.scanLora());
  // Refresh LLM models. 
  ipcMain.handle("refreshLLMModels", (event) => pathsManager.scanLLMModels());

  // Open an image with the system's default image viewer.
  ipcMain.on("openImageWithSystem", (event, url: string) => {
    let imagePath = url.replace(settings.apiHost + "/", ""); 
    imagePath = app.isPackaged 
      ? path.join(externalRes, "service", imagePath) // Path for packaged application
      : path.join(app.getAppPath(), "../service", imagePath); // Path for development 
    shell.openPath(imagePath);
  });

  // Open the system file explorer and select the image file. 
  ipcMain.on("selectImage", (event, url: string) => {
    let imagePath = url.replace(settings.apiHost + "/", ""); 
    imagePath = app.isPackaged 
      ? path.join(externalRes, "service", imagePath) 
      : path.join("..", "service", imagePath);

    if (process.platform === 'win32') {
      exec(`explorer.exe /select, "${imagePath}"`); // Use 'explorer.exe' on Windows
    } else {
      shell.showItemInFolder(imagePath); // Use the default file explorer on other platforms
    }
  });
}

/**
 * @section API Service Management
 *
 * Functions and state management for controlling the API service,
 * which runs in a separate Python process.
 */

/**
 * Object to store the state of the API service process.
 */
const apiService: {
  webProcess: ChildProcess | null; // Reference to the child process running the API service
  normalExit: boolean; // Flag indicating if the API service exited normally
} = { webProcess: null, normalExit: true };

/**
 * Check if a process with the given process ID (PID) is running. 
 * @param {number} pid - The process ID (PID) to check.
 * @returns {boolean} True if the process is running, false otherwise.
 */
function isProcessRunning(pid: number) {
  try {
    return process.kill(pid, 0); // Signal 0 is used to test for process existence without sending a signal
  } catch {
    return false;
  }
}

/**
 * Start the API service by spawning a child process running the Python script.
 * This function handles platform-specific configurations for the child process, including 
 * shell selection, Python executable path, and environment variables. 
 */
function wakeupApiService() {
  const wordkDir = path.resolve(app.isPackaged ? path.join(process.resourcesPath, "service") : path.join(__dirname, "../../../service"));
  const baseDir = app.isPackaged ? process.resourcesPath : path.join(__dirname, "../../../");

  // Determine the Python executable path based on the platform.
  const pythonExe = process.platform === 'win32'
    ? path.resolve(path.join(baseDir, "env/python.exe")) 
    : path.resolve(path.join(baseDir, "env/bin/python3")); 

  // Define environment variables for the API service.
  const newEnv = {
    "SYCL_ENABLE_DEFAULT_CONTEXTS": "1",
    "SYCL_CACHE_PERSISTENT": "1",
    "PYTHONIOENCODING": "utf-8"
  };

  // Merge the API service environment variables with the existing process environment.
  const mergedEnv = { ...process.env, ...newEnv };

  // Set up options for spawning the child process.
  const options = {
    cwd: wordkDir, // Working directory for the child process.
    detached: settings.debug, // Detach the process from the main Electron process in debug mode.
    windowsHide: !settings.debug, // Hide the process window in production mode.
    env: mergedEnv // Set the environment variables for the child process. 
  };

  // Determine the shell and arguments based on the platform and debug settings.
  const isWindows = process.platform === 'win32';
  const shell = isWindows 
    ? (settings.debug ? 'cmd.exe' : settings.windowsShell) // Use the configured shell on Windows in non-debug mode.
    : '/bin/sh'; 
  const shellArgs = isWindows ? ['/c', pythonExe, 'web_api.py'] : ['-c', `${pythonExe} web_api.py`];

  try {
    // Spawn the child process.
    apiService.webProcess = spawn(shell, shellArgs, options); 
    // Set up error handling for the child process.
    apiService.webProcess.on('error', (err) => logger.error('Failed to start subprocess:', err));
  } catch (err) {
    logger.error('Error spawning process:', err); // Log any errors during process spawning.
  }
}

/**
 * Gracefully shutdown the API service.
 * Attempts to terminate the API service cleanly by sending a shutdown request and then killing the process if necessary.
 */
function closeApiService() {
  apiService.normalExit = true;

  // Check if the API service process is running
  if (apiService.webProcess && apiService.webProcess.pid && isProcessRunning(apiService.webProcess.pid)) {
    // Attempt to terminate the process
    apiService.webProcess.kill(); 
    apiService.webProcess = null;
  }

  // Send a request to the API service's shutdown endpoint
  return fetch(`${settings.apiHost}/api/applicationExit`)
    .then(response => {
      // Handle the API shutdown response
      if (!response.ok) {
        logger.error(`API shutdown failed: ${response.status} - ${response.statusText}`); 
      } else {
        logger.info("API service shutdown successfully."); 
      }
    })
    .catch(error => {
      logger.error('Error during API exit:', error); // Log any errors that occur during the API shutdown
    });
}

/**
 * Open an image in a new window. This function is triggered by an IPC event from the renderer process. 
 * @param {IpcMainEvent} _ - The IPC event object.
 * @param {string} url - The URL of the image to open.
 * @param {string} title - The title to set for the new window.
 * @param {number} width - The initial width of the window.
 * @param {number} height - The initial height of the window.
 */
ipcMain.on("openImageWin", (_: IpcMainEvent, url: string, title: string, width: number, height: number) => {
  const display = screen.getPrimaryDisplay();
  width += 32; // Add padding to account for window chrome
  height += 48; 
  width = Math.min(width, display.workAreaSize.width); // Ensure the window is within screen bounds
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
  imgWin.setMenu(null); // Remove the menu bar
  imgWin.loadURL(url); // Load the image URL into the window
  // Show the window once it's ready
  imgWin.once("ready-to-show", () => {
    imgWin.show();
    imgWin.setTitle(title); 
  });
});

/**
 * Handle requests to show a save file dialog.
 * @param {Electron.SaveDialogOptions} options - Options to configure the save dialog.
 * @returns {Promise<Electron.SaveDialogReturnValue>} - The result of the save dialog. 
 */
ipcMain.handle('showSaveDialog', async (event, options: Electron.SaveDialogOptions) => {
  return dialog.showSaveDialog(options).catch(err => logger.error('Error in showSaveDialog:', err));
});

/**
 * Determine if the application needs to be run with admin/root privileges. 
 * This is achieved by attempting to write a file to a protected location on Windows or by checking
 * write permissions to a specific directory on Linux/macOS. 
 * 
 * @returns {Promise<boolean>} - Resolves to `true` if admin privileges are required; otherwise, `false`. 
 */
function needAdminPermission(): Promise<boolean> {
  return new Promise<boolean>((resolve, reject) => {
    if (process.platform === 'win32') {
      // Windows-specific check: Attempt to write a file to a directory that usually requires admin privileges. 
      const filename = path.join(externalRes, `${randomUUID()}.txt`);
      fs.writeFile(filename, '', (err) => {
        if (err) {
          if (err.code === 'EPERM' && path.parse(externalRes).root === path.parse(process.env.windir!).root) {
            resolve(true); // Admin privileges are needed.
          } else {
            logger.error('Failed to write file for admin check:', err); 
            resolve(false); 
          }
        } else {
          fs.rmSync(filename); // Delete the temporary file
          resolve(false); // Admin privileges are not needed.
        }
      });
    } else {
      // Linux/macOS check: Use 'fs.access' to check for write permissions on the `externalRes` directory.
      fs.access(externalRes, fs.constants.W_OK, (err) => {
        if (err) {
          logger.error('Failed to access external resource directory:', err);
          reject(err); 
        } else {
          resolve(false); 
        }
      });
    }
  });
}

/**
 * Check if the application is running with administrator/root privileges.
 * @returns {boolean} `true` if the application has admin privileges; otherwise, `false`.
 */
function isAdmin(): boolean {
  if (process.platform === 'win32') {
    // Windows-specific: Use 'koffi' to access the `IsUserAnAdmin` function from 'Shell32.dll'.
    const lib = koffi.load("Shell32.dll");
    try {
      const IsUserAnAdmin = lib.func("IsUserAnAdmin", "bool", []);
      return IsUserAnAdmin();
    } finally {
      lib.unload();
    }
  } else {
    // Linux/macOS: Check if the effective user ID is 0 (root).
    return process.getuid() === 0;
  }
}

/**
 * Start the Electron application. 
 * This function handles the initialization process, including admin privilege checks, 
 * preventing multiple instances, loading settings, setting up the window, and starting 
 * the API service. 
 */
app.whenReady().then(async () => {
  try {
    // Check if admin permissions are required and attempt to relaunch with admin rights if needed.
    if (await needAdminPermission()) {
      if (singleLock) app.releaseSingleInstanceLock(); // Release the single instance lock before restarting.
      // Execute the current command with elevated privileges using sudo-prompt.
      sudo.exec(process.argv.join(' ').trim(), (err, stdout, stderr) => {
        if (err) {
          logger.error("Sudo exec error:", err); // Log any errors during privilege elevation.
        }
        app.exit(0); // Exit the current application instance after attempting to relaunch with elevated privileges.
      });
      return;
    }

    // Prevent multiple instances of the application from running.
    if (!singleLock) {
      // Display an error message if another instance is detected.
      dialog.showMessageBoxSync({
        message: app.getLocale() === "zh-CN" ? "本程序仅允许单实例运行，确认后本次运行将自动结束" : "This program only allows a single instance to run, and the run will automatically end after confirmation",
        title: "error",
        type: "error"
      });
      app.exit(); // Exit the new instance of the application. 
    } else {
      // Proceed with application initialization
      loadSettings(); // Load the application settings.
      initEventHandle(); // Initialize all IPC and event handlers.
      createWindow(); // Create the main application window.
      wakeupApiService(); // Start the API service. 
    }
  } catch (error) {
    logger.error("Error during app initialization:", error); // Log any errors during application initialization.
  }
});
