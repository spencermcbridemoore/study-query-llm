# Google Colab Setup Guide

This guide explains how to run Study Query LLM in Google Colab.

## Quick Start

1. **Open the Colab notebook:**
   - Open `notebooks/colab_setup.ipynb` in Google Colab
   - Or upload it to your Google Drive and open with Colab

2. **Set your API keys:**
   - Edit the configuration cell with your Azure OpenAI credentials
   - Or add OpenAI/Hyperbolic keys if using those providers

3. **Run all cells:**
   - Click "Runtime" → "Run all"
   - Or run each cell sequentially

4. **Access the app:**
   - The notebook will display a Colab proxy URL (looks like `https://xxxx-5006.proxy.googleusercontent.com`)
   - Click that link to open the application in a new tab (do **not** use `http://127.0.0.1:5006`)

## Detailed Steps

### Step 1: Install Dependencies

The first cell installs all required Python packages:
- Panel (web framework)
- `jupyter_bokeh` (enables Panel widgets inside Colab)
- OpenAI SDK (for Azure and OpenAI)
- SQLAlchemy (database)
- Pandas (analytics)
- Other dependencies

### Step 2: Add Source Code

You have two options:

**Option A: Upload Project Folder**
1. Upload the entire `study-query-llm` folder to Colab
2. Uncomment the path configuration cell
3. Modify the path to match your upload location

**Option B: Clone from GitHub**
1. Uncomment the git clone commands
2. Replace the repository URL with your repo
3. The notebook will clone and install the package

### Step 3: Configure API Keys

**Recommended Method: Use Colab Secrets** (More Secure)

1. Click the 🔑 (key) icon in the left sidebar
2. Click "+ Add secret"
3. Add secrets with these exact names:
   - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
   - `AZURE_OPENAI_ENDPOINT`: Your Azure endpoint URL
   - `AZURE_OPENAI_DEPLOYMENT`: Your deployment name (e.g., "gpt-4o")
   - `AZURE_OPENAI_API_VERSION`: API version (default: "2024-02-15-preview")
   - (Optional) `OPENAI_API_KEY`: For OpenAI provider
   - (Optional) `OPENAI_MODEL`: OpenAI model name
   - (Optional) `HYPERBOLIC_API_KEY`: For Hyperbolic provider
   - (Optional) `HYPERBOLIC_ENDPOINT`: Hyperbolic endpoint URL

The notebook will automatically load these secrets. This is more secure than hardcoding keys in the notebook.

**Alternative Method: Direct Environment Variables**

If you prefer, you can set API keys directly in the code cell (less secure, but works):
- Edit the configuration cell
- Replace placeholder values with your actual API keys

### Step 4: Initialize Database

The database initialization cell creates the SQLite database and tables. This happens automatically.

### Step 5: Start Application

The final cell starts the Panel server and prints a **Colab proxy link** that tunnels traffic from your browser to the Colab runtime. Use that link (or the inline button) to open the UI—local URLs such as `http://127.0.0.1:5006` only work when you run the app on your own machine.

## Important Notes

### Session Persistence
- The Colab session ends when you close the notebook
- All data (including the database) is lost when the session ends
- To persist data, download the database file before closing

### Resource Limits
- Colab free tier has resource limits
- Long-running sessions may timeout
- Consider upgrading to Colab Pro for longer sessions

### Public URLs
- Colab creates public URLs for your app
- Anyone with the URL can access your app
- Don't share the URL publicly if you have sensitive data
- API keys are stored in environment variables (not in the URL)

## Troubleshooting

### App Doesn't Start
1. Check that all cells ran successfully
2. Verify API keys are set correctly
3. Ensure the source code is accessible (uploaded or cloned)
4. Check the error messages in the cell output
5. If you see a warning about `jupyter_bokeh`, run `!pip install jupyter_bokeh` (the notebook does this automatically, but rerun Step 1 if you interrupted it)

### Can't Access the URL
1. Make sure the server cell is still running
2. Use the Colab proxy link that looks like `https://<session>-5006.proxy.googleusercontent.com` (not `http://127.0.0.1:5006`)
3. Try refreshing the link
4. Check that Colab hasn't disconnected
5. Restart the runtime and run all cells again

### Blank Page at 127.0.0.1
- This happens when you try to open the Panel server through your local loopback address while the app is actually running inside Colab.
- Always open the proxy link printed by the notebook. It forwards traffic through Colab's infrastructure and also enables the WebSocket connection that Panel/Bokeh needs.

### Database Errors
1. The database is created automatically
2. If you see errors, try restarting the runtime
3. The database file is in the Colab session directory

### Import Errors
1. Make sure the source code is in the Python path
2. Verify you've uploaded/cloned the project correctly
3. Check that all dependencies installed successfully

## Alternative: Inline Display

If the URL method doesn't work, you can display the app inline in the notebook:

```python
from panel_app.app import create_app

app = create_app()
app.servable()
app  # Display in notebook cell
```

## Downloading Data

To save your inference data:

```python
# Download the database file
from google.colab import files
files.download('study_query_llm.db')
```

## Next Steps

Once the app is running:
1. Go to the **Inference** tab
2. Select your provider
3. For Azure: Load and select a deployment
4. Run inferences
5. Check the **Analytics** tab for results

## Limitations

- **Session-based**: Data is lost when session ends
- **Resource limits**: Free tier has CPU/RAM limits
- **Timeout**: Long idle sessions may timeout
- **Public URLs**: URLs are publicly accessible

For production use, consider:
- Docker deployment
- Cloud platform deployment (Railway, Render)
- Self-hosted server

