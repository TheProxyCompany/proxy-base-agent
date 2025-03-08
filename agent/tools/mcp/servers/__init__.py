import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from github import Github, GithubException

logger = logging.getLogger(__name__)

default_mcp_servers: dict[str, dict[str, Any]] = {
    "mcp-server-time": {
        "command": "uvx",
        "args": ["mcp-server-time"],
    },
    # Example configuration for puppeteer server
    # "puppeteer": {
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    # }
}


def download_server_from_github(
    repo_path: str,
    target_folder: str | None = None,
    branch: str = "main",
) -> bool:
    """
    Downloads all files from a GitHub repository folder to a local directory.

    Args:
        repo_path: Path to the repository folder.  Can be a URL or an owner/repo path.
        target_folder: Local folder to save files. Defaults to last part of repo_path.
        branch: Repository branch. Defaults to "main".  Overridden if in URL.

    Returns:
        True if successful, False otherwise
    """
    owner, repo_name, folder_path, branch = _parse_repo_path(repo_path, branch)
    if not all([owner, repo_name]):
        return False

    target_folder = target_folder or (
        folder_path.split("/")[-1] if folder_path else repo_name
    )
    target_path = Path(__file__).parent / target_folder

    return _download_files(owner, repo_name, folder_path, branch, target_path)

def _parse_repo_path(repo_path: str, branch: str) -> tuple[str, str, str, str]:
    """Parses the repository path and extracts owner, repo name, folder path, and branch."""
    is_url = repo_path.startswith(("http://", "https://"))
    if is_url:
        parsed = urlparse(repo_path)
        if not parsed.netloc.endswith("github.com"):
            logger.error(f"URL must be from github.com, got: {parsed.netloc}")
            return "", "", "", ""
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            logger.error(f"Invalid GitHub URL format, got: {repo_path}")
            return "", "", "", ""
        owner, repo_name = parts[0], parts[1]
        if len(parts) > 3 and parts[2] in ("tree", "blob"):
            branch = parts[3]
            folder_path = "/".join(parts[4:]) if len(parts) > 4 else ""
        else:
            folder_path = ""
    else:
        parts = repo_path.split("/")
        if len(parts) < 2:
            logger.error(f"Invalid repository path format: '{repo_path}'")
            return "", "", "", ""
        owner, repo_name = parts[0], parts[1]
        folder_path = "/".join(parts[2:]) if len(parts) > 2 else ""

    return owner, repo_name, folder_path, branch

def _download_files(
    owner: str,
    repo_name: str,
    folder_path: str,
    branch: str,
    target_path: Path,
) -> bool:
    """Downloads files from a GitHub repository folder to a local directory."""
    token = os.environ.get("GITHUB_TOKEN")
    g = Github(token) if token else Github()
    logger.info(
        "Using GitHub token for authentication"
        if token
        else "No GitHub token found. Using unauthenticated access (rate limits may apply)"
    )

    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
        contents = repo.get_contents(folder_path, ref=branch)
        if not isinstance(contents, list):
            contents = [contents]

        target_path.mkdir(exist_ok=True)
        logger.debug(f"Downloading files from {owner}/{repo_name}/{folder_path} to {target_path}")

        to_process = contents.copy()
        while to_process:
            content = to_process.pop()
            if content.type == "dir":
                sub_contents = repo.get_contents(content.path, ref=branch)
                to_process.extend(
                    sub_contents if isinstance(sub_contents, list) else [sub_contents]
                )
                sub_path = content.path.replace(folder_path, "", 1).lstrip("/")
                if sub_path:
                    (target_path / sub_path).mkdir(exist_ok=True)
            else:
                try:
                    file_bytes = content.decoded_content
                    rel_path = content.path.replace(folder_path, "", 1).lstrip("/")
                    file_path = (
                        target_path / rel_path
                        if rel_path
                        else target_path / content.name
                    )
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(file_bytes)
                    logger.debug(f"Downloaded: {file_path}")
                except (OSError, GithubException) as e:
                    logger.error(f"Error downloading {content.path}: {e}")

        logger.debug(
            f"Successfully downloaded all files from {owner}/{repo_name}/{folder_path} to {target_path}"
        )
        return True
    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        if e.status == 404:
            logger.warning(
                "Repository or folder not found. Check the path and your access permissions."
            )
        elif e.status == 403:
            logger.warning(
                "Rate limit exceeded or authentication required. Try using a GitHub token."
            )
        return False
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    # Test with GitHub URL
    url_path = "https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer"
    logger.info(f"Testing with GitHub URL: {url_path}")
    success = download_server_from_github(url_path)

    # # Test with URL containing encoded characters
    # if success:
    #     logger.info("\nTesting with URL containing encoded characters")
    #     encoded_url = (
    #         "https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search"
    #     )
    #     success = download_server_from_github(encoded_url)
