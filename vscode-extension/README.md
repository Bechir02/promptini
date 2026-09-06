# Prompt Forge VS Code Extension

This extension launches the local Prompt Forge Gradio app and opens it inside a VS Code Webview.

## Features

- 🚀 **Dynamic Port Selection** — Automatically finds an available port if 7860 is in use
- ⏹️ **Explicit Stop Command** — Stop the server with a single command
- 📦 **.vsix Packaging** — Build a distributable extension package

## Setup

```bash
cd vscode-extension
npm install
npm run compile
```

## Usage

### From VS Code

1. Open the repository in VS Code
2. Press `F5` to start the Extension Development Host
3. Run the command `Prompt Forge: Open App`
4. Optionally, run `Prompt Forge: Stop App` to stop the server

### From Command Palette

- `Prompt Forge: Open App` — Start the server and open in webview
- `Prompt Forge: Stop App` — Stop the running server

### Build a Distributable .vsix

```bash
cd vscode-extension
npm install
npm run compile
npm run package
```

This generates `prompt-forge-vscode-0.1.0.vsix` which can be installed directly into VS Code or shared.

## Requirements

- A Python interpreter available as `python3` on macOS/Linux or `python` on Windows
- `app.py` in the workspace root
- `npm` installed for compiling the extension

## How It Works

1. Extension detects the first available port starting from 7860
2. Launches `app.py` with `GRADIO_SERVER_PORT` environment variable
3. Waits for the server to become available
4. Opens the app inside a VS Code Webview panel
5. Manages the Python process lifecycle
