import shutil
import subprocess
import sys


def ensure_npx_exists():
    if shutil.which("npx") is None:
        print("Error: 'npx' is not installed or not found in PATH.")
        print(
            "Install Node.js from https://nodejs.org and ensure 'npx' is in your PATH."
        )
        sys.exit(1)

def run_npx(args: list[str] | None = None):
    ensure_npx_exists()
    command = ["npx"] + (args or [])
    subprocess.run(command, check=True)

default_mcp_servers = {
    "mcp-server-time": {
        "command": "uvx",
        "args": ["mcp-server-time"],
    },
    "everything": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everything"],
    }
}
