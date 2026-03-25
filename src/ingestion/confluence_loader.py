from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


class ConfluenceLoader:
    def __init__(self, base_url: Optional[str] = None, email: Optional[str] = None, api_token: Optional[str] = None):
        self.base_url = (base_url or os.getenv("CONFLUENCE_BASE_URL", "")).rstrip("/")
        self.email = email or os.getenv("CONFLUENCE_EMAIL", "")
        self.api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN", "")
        self.session = requests.Session()
        if self.email and self.api_token:
            self.session.auth = (self.email, self.api_token)
        self.session.headers.update({"Accept": "application/json"})

    def list_pages_in_space(self, space_key: str, limit: int = 25) -> List[Dict]:
        start = 0
        all_pages: List[Dict] = []
        while True:
            resp = self.session.get(
                f"{self.base_url}/rest/api/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "limit": limit,
                    "start": start,
                    "expand": "version,history",
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("results", [])
            all_pages.extend(batch)
            if len(batch) < limit:
                break
            start += limit
        return all_pages

    def get_page(self, page_id: str) -> Dict:
        resp = self.session.get(
            f"{self.base_url}/rest/api/content/{page_id}",
            params={"expand": "body.storage,version,space"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text("\n").splitlines()]
        return "\n".join([line for line in lines if line])
