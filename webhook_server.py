import os
import requests
from flask import Flask, request

app = Flask(__name__)

GITHUB_USERNAME = "burhanuddinkhandwala786"
REPO_NAME = "catalog-search-suite"
WORKFLOW_FILE = "auto_sync.yml"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

def trigger_github_workflow():
    """Calls GitHub API to launch the auto-sync runner instantly."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    data = {"ref": "main"}
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 204:
        print("⚡ Successfully triggered GitHub Instant Cloud Sync!")
    else:
        print(f"❌ Failed to trigger GitHub Action: {response.status_code} - {response.text}")

# Root route so Render's health checker gets an instant 200 OK response
@app.route("/", methods=["GET", "HEAD"])
def health_check():
    return "Webhook Listener Active", 200

@app.route("/drive-webhook", methods=["POST"])
def drive_webhook():
    resource_state = request.headers.get("X-Goog-Resource-State")
    
    if resource_state in ["add", "update", "sync"]:
        trigger_github_workflow()
        
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
