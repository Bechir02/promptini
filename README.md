---
title: Prompt Forge Rag
emoji: 😻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## VS Code Extension Wrapper

A full-featured extension scaffold is available in `vscode-extension/`.

### Features

- 🚀 **Dynamic Port Selection** — Automatically finds an available port if 7860 is occupied
- ⏹️ **Explicit Stop Command** — Cleanly stop the server without closing the panel
- 📦 **.vsix Packaging** — Build a distributable extension for team distribution

### Quick Start

```bash
cd vscode-extension
npm install
npm run compile
```

Then:
1. Press `F5` in VS Code to launch the Extension Development Host
2. Run the command `Prompt Forge: Open App`
3. To stop: Run `Prompt Forge: Stop App`

### Build a Distributable Package

```bash
npm run package
```

This creates `prompt-forge-vscode-0.1.0.vsix` which can be shared or installed via:
```bash
code --install-extension prompt-forge-vscode-0.1.0.vsix
```

### How It Works

- Detects and uses the first available port starting from 7860
- Spawns `app.py` with `GRADIO_SERVER_PORT` environment variable
- Waits for the Gradio server to boot
- Embeds the UI in a lightweight VS Code Webview
- Manages the Python process lifecycle automatically

For detailed information, see [vscode-extension/README.md](vscode-extension/README.md).
