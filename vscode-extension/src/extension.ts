import * as http from "http";
import * as net from "net";
import * as path from "path";
import * as fs from "fs";
import * as vscode from "vscode";
import { ChildProcess, spawn } from "child_process";

let promptForgeProcess: ChildProcess | undefined;
let currentPort: number | undefined;
let currentPanel: vscode.WebviewPanel | undefined;

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

function getWebviewHtml(port: number): string {
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
        background: #f8fafc;
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
        padding: 12px 16px;
        background: rgba(15, 23, 42, 0.92);
        color: white;
        font-family: sans-serif;
        font-size: 0.95rem;
        z-index: 10;
      }
      .iframe-container {
        position: absolute;
        top: 42px;
        left: 0;
        right: 0;
        bottom: 0;
      }
    </style>
  </head>
  <body>
    <div class="banner">Prompt Forge is running on port ${port}. Reload the panel if needed.</div>
    <div class="iframe-container">
      <iframe src="http://127.0.0.1:${port}/"></iframe>
    </div>
  </body>
</html>`;
}

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

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel("Prompt Forge Server");

  const openCommand = vscode.commands.registerCommand("promptForge.open", async () => {
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

    // Kill any existing process before starting a new one
    if (promptForgeProcess && !promptForgeProcess.killed) {
      stopPromptForge(output);
    }

    output.show(true);
    output.appendLine("Starting Prompt Forge Python server...");

    // Find an available port
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
    if (currentPanel) {
      currentPanel.dispose();
    }

    currentPanel = vscode.window.createWebviewPanel("promptForge", "Prompt Forge", vscode.ViewColumn.One, {
      enableScripts: true,
      retainContextWhenHidden: true,
    });
    currentPanel.webview.html = getWebviewHtml(port);

    currentPanel.onDidDispose(() => {
      currentPanel = undefined;
    });

    vscode.window.showInformationMessage(`Prompt Forge is ready on port ${port}`);
  });

  const stopCommand = vscode.commands.registerCommand("promptForge.stop", async () => {
    stopPromptForge(output);
    output.appendLine("Prompt Forge stopped.");
    vscode.window.showInformationMessage("Prompt Forge has been stopped.");
  });

  context.subscriptions.push(openCommand, stopCommand);
}

export function deactivate() {
  if (promptForgeProcess && !promptForgeProcess.killed) {
    promptForgeProcess.kill();
    promptForgeProcess = undefined;
  }
}
