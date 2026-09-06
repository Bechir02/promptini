# VS Code Extension Setup Guide

## All Requirements

### Node.js & npm
- Node.js 14.x or higher
- npm (included with Node.js)

### TypeScript Compiler
- Installed automatically via `npm install`

### VS Code
- Version 1.88.0 or later
- Extension Development Host (`F5` debug mode)

### Python
- Python 3.7+
- Available as `python3` (macOS/Linux) or `python` (Windows)

### App Files
- `app.py` must exist in the workspace root
- All Python dependencies from `requirements.txt` must be installed

---

## Step-by-Step Setup

### 1. Install Node Dependencies

```bash
cd vscode-extension
npm install
```

This installs:
- `typescript` — TypeScript compiler
- `@types/vscode` — VS Code type definitions
- `@types/node` — Node.js type definitions
- `@vscode/vsce` — Packaging tool for .vsix

### 2. Compile TypeScript

```bash
npm run compile
```

This generates the `out/extension.js` file from `src/extension.ts`.

### 3. Run in Development Mode

```bash
# From the root of the repository
code .
```

Then inside VS Code:
- Press `F5` (or use Debug menu)
- This opens "Extension Development Host" — a new VS Code window with the extension active

### 4. Use the Commands

In the Extension Development Host:
- Press `Cmd+Shift+P` (or `Ctrl+Shift+P` on Windows/Linux)
- Type `Promptini: Open App`
- The server will start, and the UI opens in the Webview

To stop:
- Press `Cmd+Shift+P`
- Type `Promptini: Stop App`

---

## Build a Distributable .vsix Package

```bash
npm run package
```

This creates:
- `prompt-forge-vscode-0.1.0.vsix` in the `vscode-extension` folder

### Install the .vsix Package Locally

```bash
code --install-extension prompt-forge-vscode-0.1.0.vsix
```

Or:
1. Open VS Code
2. Press `Cmd+Shift+X` (Extensions view)
3. Click the three-dot menu → "Install from VSIX..."
4. Select the `.vsix` file

### Share the .vsix Package

You can distribute `prompt-forge-vscode-0.1.0.vsix` to teammates. They can install it the same way.

---

## Features Explained

### Dynamic Port Selection
If port 7860 is already in use (by another Gradio app or service), the extension automatically tries ports 7861, 7862, etc. The banner shows which port is being used.

### Explicit Stop Command
- `Promptini: Stop App` kills the Python process and closes the Webview
- Useful if you want to restart the app or free up resources

### Environment Variable Injection
The extension passes `GRADIO_SERVER_PORT` to the Python process, ensuring Gradio listens on the selected port.

---

## Troubleshooting

### "Could not find app.py"
- Ensure you opened the repository folder in VS Code (not a subfolder)
- Verify `app.py` exists in the workspace root

### "Server did not become available"
- Check the "Promptini Server" output channel for Python errors
- Verify all Python dependencies are installed: `pip install -r requirements.txt`
- Check if port 7860+ are available: `lsof -i :7860` (macOS/Linux)

### Extension doesn't activate
- Ensure you pressed `F5` from the Extension Development Host window
- Check the Debug Console for error messages

### .vsix package failed
- Run `npm run compile` first to generate `out/extension.js`
- Ensure `@vscode/vsce` is installed: `npm install`

---

## Next Steps

- Share the `.vsix` with your team
- Customize the extension further (e.g., add keybindings, settings)
- Publish to the VS Code Marketplace (requires a publisher account)
