from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from github import Github, GithubException

logger = logging.getLogger(__name__)

@dataclass
class MCPServer:
    name: str
    description: str
    vendor: str
    sourceUrl: str
    homepage: str
    license: str
    runtime: str

    @classmethod
    def load_available_servers_from_json(
        cls, json_file_path: str | None = None
    ) -> list[MCPServer]:
        """
        Load server definitions from the server_list.json file and return as list of MCPServer objects.

        Args:
            json_file_path: Optional path to the JSON file. If not provided, will use the default location.

        Returns:
            List of MCPServer objects loaded from the JSON file
        """
        if json_file_path is None:
            json_file_path = str(Path(__file__).parent / "servers/servers_list.json")

        try:
            with open(json_file_path) as f:
                server_data = json.load(f)

            return [cls(**server) for server in server_data]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading server list from {json_file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading server list: {e}")
            return []

    @staticmethod
    def download_server_from_github(
        repo_path: str,
        target_folder: str | None = None,
        branch: str = "main",
    ) -> bool:
        """
        Downloads all files from a GitHub repository folder to a local directory.

        Args:
            repo_path: Path to the repository folder. Can be a URL or an owner/repo path.
            target_folder: Local folder to save files. Defaults to last part of repo_path.
            branch: Repository branch. Defaults to "main". Overridden if in URL.

        Returns:
            True if successful, False otherwise
        """
        owner, repo_name, folder_path, branch = MCPServer._parse_repo_path(
            repo_path, branch
        )
        if not all([owner, repo_name]):
            return False

        target_folder = target_folder or (
            folder_path.split("/")[-1] if folder_path else repo_name
        )
        target_path = Path(__file__).parent / "servers" / target_folder

        return MCPServer._download_files(
            owner, repo_name, folder_path, branch, target_path
        )

    @staticmethod
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

    @staticmethod
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
            logger.debug(
                f"Downloading files from {owner}/{repo_name}/{folder_path} to {target_path}"
            )

            to_process = contents.copy()
            while to_process:
                content = to_process.pop()
                if content.type == "dir":
                    sub_contents = repo.get_contents(content.path, ref=branch)
                    to_process.extend(
                        sub_contents
                        if isinstance(sub_contents, list)
                        else [sub_contents]
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
