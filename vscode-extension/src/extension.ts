import * as http from "http";
import * as net from "net";
import * as path from "path";
import * as fs from "fs";
import * as vscode from "vscode";
import { ChildProcess, spawn } from "child_process";

let promptForgeProcess: ChildProcess | undefined;
let currentPort: number | undefined;
let currentPanel: vscode.WebviewPanel | undefined;

// ── Prompt Library (persisted in globalState) ────────────────────────────────
interface LibraryEntry {
  id: string;
  prompt: string;
  model: string;
  status: string;
  date: string;
}

function getLibrary(ctx: vscode.ExtensionContext): LibraryEntry[] {
  return ctx.globalState.get<LibraryEntry[]>("promptForgeLibrary", []);
}

function saveLibrary(ctx: vscode.ExtensionContext, lib: LibraryEntry[]) {
  ctx.globalState.update("promptForgeLibrary", lib);
}

// ── Utility functions ────────────────────────────────────────────────────────
function canFileExist(filePath: string): Promise<boolean> {
  return new Promise((resolve) => {
    fs.access(filePath, fs.constants.F_OK, (err) => {
      resolve(!err);
    });
  });
}

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close();
      resolve(true);
    });
    server.listen(port);
  });
}

async function findAvailablePort(startPort: number = 7860): Promise<number> {
  for (let port = startPort; port < startPort + 100; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error("Could not find an available port");
}

function waitForServer(port: number, timeoutMs: number): Promise<void> {
  const end = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const check = () => {
      const request = http.request({ host: "127.0.0.1", port, method: "GET", path: "/" }, (res) => {
        res.destroy();
        resolve();
      });

      request.on("error", () => {
        if (Date.now() >= end) {
          reject(new Error(`Server did not become available on port ${port} in ${timeoutMs}ms`));
        } else {
          setTimeout(check, 500);
        }
      });

      request.setTimeout(2000, () => {
        request.destroy();
      });
      request.end();
    };
    check();
  });
}

// ── Webview HTML with message bridge ─────────────────────────────────────────
function getWebviewHtml(url: string, mode: string): string {
  const banner = mode === "local" 
    ? `<div class="banner">Local Server Running. Port: ${new URL(url).port}</div>`
    : "";
  const containerTop = mode === "local" ? "42px" : "0";

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Prompt Forge</title>
    <style>
      body, html {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: white;
      }
      iframe {
        border: none;
        width: 100%;
        height: 100%;
      }
      .banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        padding: 10px 16px;
        background: #c4633e;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 12px;
        font-weight: 600;
        z-index: 10;
        text-align: center;
      }
      .iframe-container {
        position: absolute;
        top: ${containerTop};
        left: 0;
        right: 0;
        bottom: 0;
      }
    </style>
  </head>
  <body>
    ${banner}
    <div class="iframe-container">
      <iframe id="app-frame" src="${url}"></iframe>
    </div>
    <script>
      const vscode = acquireVsCodeApi();

      // Listen for messages FROM the Gradio iframe (via window.top.postMessage)
      window.addEventListener('message', (event) => {
        const data = event.data;
        if (!data || !data.type) return;

        // Forward to the VS Code extension host
        if (data.type === 'copyText' || data.type === 'savePrompt' || data.type === 'ready') {
          vscode.postMessage(data);
        }
      });

      // Listen for messages FROM the VS Code extension host
      window.addEventListener('message', (event) => {
        const data = event.data;
        if (!data || !data.type) return;

        // Forward to the Gradio iframe
        if (data.type === 'syncLibrary' || data.type === 'setPrompt') {
          const iframe = document.getElementById('app-frame');
          if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage(data, '*');
          }
        }
      });
    </script>
  </body>
</html>`;
}

// ── Python launcher ──────────────────────────────────────────────────────────
function choosePythonCommand(): string {
    const venvPython = "/Users/mac/Desktop/prompt-forge-rag/.venv/bin/python3";
    if (require("fs").existsSync(venvPython)) {
        return venvPython;
    }
    return process.platform === "win32" ? "python" : "python3";
}

function launchPromptForge(appPath: string, cwd: string, port: number, output: vscode.OutputChannel): ChildProcess {
  const python = choosePythonCommand();
  const env = { ...process.env, GRADIO_SERVER_PORT: port.toString() };
  const childProcess = spawn(python, [appPath], {
    cwd,
    env,
    shell: false,
  });

  childProcess.stdout?.on("data", (chunk: Buffer) => {
    output.append(chunk.toString());
  });

  childProcess.stderr?.on("data", (chunk: Buffer) => {
    output.append(chunk.toString());
  });

  childProcess.on("exit", (code: number | null, signal: NodeJS.Signals | null) => {
    output.appendLine(`Prompt Forge process exited with code=${code} signal=${signal}`);
    promptForgeProcess = undefined;
  });

  return childProcess;
}

function stopPromptForge(output: vscode.OutputChannel) {
  if (promptForgeProcess && !promptForgeProcess.killed) {
    output.appendLine("Stopping Prompt Forge...");
    promptForgeProcess.kill();
    promptForgeProcess = undefined;
    currentPort = undefined;
  }
  if (currentPanel) {
    currentPanel.dispose();
    currentPanel = undefined;
  }
}

// ── Message handler (shared between sidebar and panel) ───────────────────────
function setupMessageHandler(
  webview: vscode.Webview,
  context: vscode.ExtensionContext
) {
  webview.onDidReceiveMessage((message: any) => {
    switch (message.type) {
      case "copyText": {
        if (message.text) {
          vscode.env.clipboard.writeText(message.text).then(() => {
            vscode.window.showInformationMessage("📋 Prompt copied to clipboard!");
          });
        }
        break;
      }
      case "savePrompt": {
        if (message.entry) {
          const lib = getLibrary(context);
          const entry: LibraryEntry = {
            id: Date.now().toString(),
            prompt: message.entry.prompt || "",
            model: message.entry.model || "unknown",
            status: message.entry.status || "",
            date: new Date().toISOString(),
          };
          lib.unshift(entry);
          saveLibrary(context, lib);
          vscode.window.showInformationMessage("⭐ Prompt saved to library!");
          // Sync back to webview
          webview.postMessage({ type: "syncLibrary", library: lib });
        }
        break;
      }
      case "ready": {
        const lib = getLibrary(context);
        webview.postMessage({ type: "syncLibrary", library: lib });
        break;
      }
    }
  });
}

// ── Sidebar WebviewViewProvider ──────────────────────────────────────────────
class PromptForgeSidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "promptForgeView";
  private _view?: vscode.WebviewView;

  constructor(private readonly _context: vscode.ExtensionContext) {}

  resolveWebviewView(webviewView: vscode.WebviewView) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
    };

    const config = vscode.workspace.getConfiguration("promptForge");
    const hfUrl = config.get<string>("hfUrl") || "https://becher-zribi-prompt-forge-rag.hf.space";
    
    webviewView.webview.html = getWebviewHtml(hfUrl, "cloud");
    setupMessageHandler(webviewView.webview, this._context);
  }

  public sendPrompt(text: string) {
    if (this._view) {
      this._view.webview.postMessage({ type: "setPrompt", text });
    }
  }
}

// ── Activation ──────────────────────────────────────────────────────────────
export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel("Prompt Forge Server");

  // Register sidebar provider
  const sidebarProvider = new PromptForgeSidebarProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      PromptForgeSidebarProvider.viewType,
      sidebarProvider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  // "Open App" command — opens in a full editor panel
  const openCommand = vscode.commands.registerCommand("promptForge.open", async () => {
    if (currentPanel) {
      currentPanel.reveal(vscode.ViewColumn.One);
    } else {
      currentPanel = vscode.window.createWebviewPanel("promptForge", "Prompt Forge", vscode.ViewColumn.One, {
        enableScripts: true,
        retainContextWhenHidden: true,
      });

      currentPanel.onDidDispose(() => {
        currentPanel = undefined;
      });
    }

    const config = vscode.workspace.getConfiguration("promptForge");
    const mode = config.get<string>("mode") || "cloud";
    const hfUrl = config.get<string>("hfUrl") || "https://huggingface.co/spaces/Becher-zribi/prompt-forge-rag";

    if (mode === "cloud") {
      output.appendLine(`Opening Cloud Mode: ${hfUrl}`);
      currentPanel.webview.html = getWebviewHtml(hfUrl, "cloud");
      setupMessageHandler(currentPanel.webview, context);
    } else {
      const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
      if (!workspaceFolder) {
        vscode.window.showErrorMessage("Open a workspace folder before launching Prompt Forge.");
        return;
      }

      const workspaceRoot = workspaceFolder.uri.fsPath;
      const appPath = path.join(workspaceRoot, "app.py");

      if (!(await canFileExist(appPath))) {
        vscode.window.showErrorMessage("Could not find app.py in the workspace root.");
        return;
      }

      if (promptForgeProcess && !promptForgeProcess.killed) {
        stopPromptForge(output);
      }

      output.show(true);
      output.appendLine("Starting Prompt Forge Python server...");

      let port = 7860;
      try {
        port = await findAvailablePort();
        output.appendLine(`Using port ${port}`);
      } catch (err) {
        vscode.window.showErrorMessage(`Could not find an available port: ${err instanceof Error ? err.message : err}`);
        return;
      }

      promptForgeProcess = launchPromptForge(appPath, workspaceRoot, port, output);

      try {
        await waitForServer(port, 120000);
      } catch (err) {
        vscode.window.showErrorMessage(`Prompt Forge did not start in time: ${err instanceof Error ? err.message : err}`);
        return;
      }

      currentPort = port;
      currentPanel.webview.html = getWebviewHtml(`http://127.0.0.1:${port}/`, "local");
      setupMessageHandler(currentPanel.webview, context);
      vscode.window.showInformationMessage(`Prompt Forge is ready on port ${port}`);
    }
  });

  // "Forge Selection" command — sends selected text to the sidebar
  const forgeSelectionCmd = vscode.commands.registerCommand("promptForge.forgeSelection", () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const selection = editor.document.getText(editor.selection);
    if (selection) {
      sidebarProvider.sendPrompt(selection);
      vscode.window.showInformationMessage("Text sent to Prompt Forge sidebar.");
    }
  });

  const stopCommand = vscode.commands.registerCommand("promptForge.stop", async () => {
    stopPromptForge(output);
    output.appendLine("Prompt Forge stopped.");
    vscode.window.showInformationMessage("Prompt Forge has been stopped.");
  });

  context.subscriptions.push(openCommand, stopCommand, forgeSelectionCmd);
}

export function deactivate() {
  if (promptForgeProcess && !promptForgeProcess.killed) {
    promptForgeProcess.kill();
    promptForgeProcess = undefined;
  }
}
