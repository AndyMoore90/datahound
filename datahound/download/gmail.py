import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .types import DownloadConfig


class GmailDownloader:
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        # logs under data/<company>/logs/pipeline
        self.logs_root = self.data_dir.parent / "logs"
        self.pipeline_dir = self.logs_root / "pipeline"
        self.processed_ids_path = self.data_dir / "processed_message_ids.json"
        self.log_path = self.pipeline_dir / "download.jsonl"
        self.service = None
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        self._setup_gmail_service()

    def _setup_gmail_service(self) -> None:
        creds: Optional[Credentials] = None
        token_file = self.config.gmail.token_path
        creds_file = self.config.gmail.credentials_path
        if token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_file), scopes=self.config.gmail.scopes
                )
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_file.exists():
                    raise FileNotFoundError("credentials.json not found for this company")
                try:
                    with open(creds_file, "r", encoding="utf-8") as f:
                        client_config = json.load(f)
                except Exception:
                    raise ValueError("Invalid credentials.json. Ensure it is a valid OAuth client JSON.")

                if not any(k in client_config for k in ("installed", "web")):
                    raise ValueError(
                        "Invalid Google OAuth client file. Create an OAuth Client ID for 'Desktop app' "
                        "in Google Cloud Console (APIs & Services → Credentials), download the JSON, and "
                        "place it at the configured credentials_path. Service accounts are not supported for Gmail."
                    )

                flow = InstalledAppFlow.from_client_config(client_config, self.config.gmail.scopes)
                creds = flow.run_local_server(port=0)
            token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(token_file, "w", encoding="utf-8") as token:
                token.write(creds.to_json())
        self.service = build("gmail", "v1", credentials=creds)

    def _load_processed_ids(self) -> List[str]:
        if not self.processed_ids_path.exists():
            return []
        with open(self.processed_ids_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_processed_ids(self, ids: List[str]) -> None:
        with open(self.processed_ids_path, "w", encoding="utf-8") as f:
            json.dump(ids, f)

    def _log(self, record: Dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _is_desired_attachment(self, part: Dict) -> bool:
        filename = part.get("filename", "")
        filename_lower = filename.lower()
        if not filename:
            return False
        has_extension = any(filename_lower.endswith(ext.lower()) for ext in self.config.allowed_extensions)
        has_attachment_id = bool(part.get("body", {}).get("attachmentId"))
        return has_extension and has_attachment_id

    def _collect_parts(self, payload: Dict) -> List[Dict]:
        stack = [payload]
        parts: List[Dict] = []
        while stack:
            node = stack.pop()
            parts.append(node)
            for child in node.get("parts", []) or []:
                stack.append(child)
        return parts

    def _extract_html_bodies(self, payload: Dict) -> List[str]:
        html_list: List[str] = []
        for part in self._collect_parts(payload):
            if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
                data = part["body"]["data"]
                html_list.append(base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore"))
        if not html_list and payload.get("body", {}).get("data") and payload.get("mimeType") == "text/html":
            html_list.append(base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore"))
        return html_list

    def _find_links(self, html: str) -> List[str]:
        """Find links matching configured prefixes from multiple HTML sources"""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        
        # Check <a> tags with href
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(href.startswith(prefix) for prefix in self.config.gmail.link_prefixes):
                links.append(href)
        
        # Check <button> tags with onclick containing URLs
        for button in soup.find_all("button"):
            onclick = button.get("onclick", "")
            if onclick:
                url_pattern = r'https?://[^\s\'"<>]+'
                urls = re.findall(url_pattern, onclick)
                for url in urls:
                    if any(url.startswith(prefix) for prefix in self.config.gmail.link_prefixes):
                        links.append(url)
            
            # Check data attributes
            for attr in button.attrs:
                if 'url' in attr.lower() or 'href' in attr.lower():
                    val = button.get(attr, "")
                    if val and any(val.startswith(prefix) for prefix in self.config.gmail.link_prefixes):
                        links.append(val)
        
        # Check images wrapped in <a> tags
        for img in soup.find_all("img"):
            parent = img.parent
            if parent and parent.name == "a" and parent.get("href"):
                href = parent["href"]
                if any(href.startswith(prefix) for prefix in self.config.gmail.link_prefixes):
                    links.append(href)
        
        return links
    
    def _find_all_links(self, html: str) -> List[str]:
        """Find all links in HTML for debugging - checks multiple sources"""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        
        # Check <a> tags with href
        for a in soup.find_all("a", href=True):
            links.append(a["href"])
        
        # Check <button> tags with onclick or data attributes
        for button in soup.find_all("button"):
            onclick = button.get("onclick", "")
            if onclick:
                # Extract URLs from onclick JavaScript
                url_pattern = r'https?://[^\s\'"<>]+'
                urls = re.findall(url_pattern, onclick)
                links.extend(urls)
            
            # Check data attributes
            for attr in button.attrs:
                if 'url' in attr.lower() or 'href' in attr.lower():
                    val = button.get(attr, "")
                    if val and val.startswith(('http://', 'https://')):
                        links.append(val)
        
        # Check images with links (wrapped in <a>)
        for img in soup.find_all("img"):
            parent = img.parent
            if parent and parent.name == "a" and parent.get("href"):
                links.append(parent["href"])
        
        # Check iframes with src
        for iframe in soup.find_all("iframe", src=True):
            links.append(iframe["src"])
        
        # Check meta refresh redirects
        for meta in soup.find_all("meta", attrs={"http-equiv": "refresh"}):
            content = meta.get("content", "")
            if "url=" in content.lower():
                url = content.split("url=", 1)[1].strip()
                links.append(url)
        
        return links

    def _download_attachment(self, message_id: str, part: Dict) -> Optional[str]:
        original_filename = part.get("filename", "").strip()
        attachment_id = part["body"]["attachmentId"]
        attachment = self.service.users().messages().attachments().get(userId="me", messageId=message_id, id=attachment_id).execute()
        data = base64.urlsafe_b64decode(attachment.get("data", b""))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part = original_filename.rsplit(".", 1)[0]
        new_filename = f"{name_part}_{timestamp}{Path(original_filename).suffix}"
        filepath = self.data_dir / new_filename
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(data)
        return new_filename

    def _download_from_link(self, file_type: str, link: str) -> Optional[str]:
        try:
            resp = requests.get(link, timeout=30)
            if resp.status_code != 200:
                return None
        except Exception as e:
            return None
        content_type = resp.headers.get("content-type", "").lower()
        cd = resp.headers.get("content-disposition", "")
        # determine extension from content-type, content-disposition, URL, or fallback for known providers
        ext: Optional[str] = None
        # content-type hints
        if ("excel" in content_type) or ("spreadsheet" in content_type):
            ext = ".xlsx"
        elif "csv" in content_type:
            ext = ".csv"
        # URL suffix hints
        if not ext:
            low = link.lower()
            if low.endswith(".csv"):
                ext = ".csv"
            elif low.endswith(".xlsx"):
                ext = ".xlsx"
            elif low.endswith(".xls"):
                ext = ".xls"
        # Content-Disposition filename
        filename_from_cd: Optional[str] = None
        if not ext and cd:
            try:
                parts = [p.strip() for p in cd.split(";")]
                fn_star = next((p for p in parts if p.lower().startswith("filename*=")), None)
                fn = next((p for p in parts if p.lower().startswith("filename=")), None)
                if fn_star:
                    val = fn_star.split("=", 1)[1].strip().strip('"').strip("'")
                    if "''" in val:
                        val = val.split("''", 1)[1]
                    filename_from_cd = unquote(val)
                elif fn:
                    val = fn.split("=", 1)[1].strip().strip('"').strip("'")
                    filename_from_cd = val
            except Exception:
                filename_from_cd = None
            if filename_from_cd:
                lower_fn = filename_from_cd.lower()
                if lower_fn.endswith(".csv"):
                    ext = ".csv"
                elif lower_fn.endswith(".xlsx"):
                    ext = ".xlsx"
                elif lower_fn.endswith(".xls"):
                    ext = ".xls"
        # URL path basename fallback
        if not ext:
            path = urlparse(link).path.lower()
            if path.endswith(".csv"):
                ext = ".csv"
            elif path.endswith(".xlsx"):
                ext = ".xlsx"
            elif path.endswith(".xls"):
                ext = ".xls"
        # Default to .xlsx for known ServiceTitan-like links if still unknown
        if not ext and any(link.startswith(prefix) for prefix in self.config.gmail.link_prefixes):
            ext = ".xlsx"
        if not ext:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # prefer filename from Content-Disposition if present
        if filename_from_cd and Path(filename_from_cd).suffix:
            name_part = Path(filename_from_cd).stem
        else:
            name_part = file_type
        filename = f"{name_part}_{timestamp}{ext}"
        filepath = self.data_dir / filename
        with open(filepath, "wb") as f:
            f.write(resp.content)
        return filename

    def _mark_read(self, message_id: str) -> None:
        if not self.config.mark_as_read:
            return
        self.service.users().messages().modify(userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}).execute()

    def find_unread_message_ids(self, file_type: str) -> List[str]:
        query = self.config.gmail.query_by_type.get(file_type, f'subject:"{file_type}" is:unread')
        message_ids = self._execute_search(query, file_type)
        
        # If no unread messages found, also check read messages for this file type
        # This allows retrying messages that were read but not successfully downloaded
        if not message_ids:
            read_query = self.config.gmail.query_by_type.get(file_type, f'subject:"{file_type}"').replace(' is:unread', '')
            if 'is:unread' not in read_query:
                read_query = f'{read_query} -is:unread'
            read_ids = self._execute_search(read_query, file_type)
            if read_ids:
                self._log({
                    "ts": datetime.now().isoformat(),
                    "company": self.config.company,
                    "file_type": file_type,
                    "status": "searching_read_messages",
                    "query": read_query,
                    "read_messages_found": len(read_ids),
                })
            message_ids.extend(read_ids)
        
        return message_ids
    
    def _execute_search(self, query: str, file_type: str = "") -> List[str]:
        """Execute Gmail search and return message IDs"""
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "file_type": file_type,
            "status": "searching_messages",
            "query": query,
        })
        results = self.service.users().messages().list(userId="me", q=query, maxResults=100).execute()
        messages = results.get("messages", [])
        message_ids = [m["id"] for m in messages]
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "file_type": file_type,
            "status": "search_complete",
            "query": query,
            "messages_found": len(message_ids),
            "message_ids": message_ids[:10],
        })
        return message_ids

    def download_for_type(self, file_type: str) -> List[str]:
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "file_type": file_type,
            "status": "download_started",
            "allowed_extensions": self.config.allowed_extensions,
            "link_prefixes": self.config.gmail.link_prefixes,
        })
        processed_ids = set(self._load_processed_ids())
        message_ids = self.find_unread_message_ids(file_type)
        downloaded_files: List[str] = []
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "file_type": file_type,
            "status": "processing_messages",
            "total_messages": len(message_ids),
            "already_processed": len([m for m in message_ids if m in processed_ids]),
        })
        for message_id in message_ids:
            if message_id in processed_ids:
                self._log({
                    "ts": datetime.now().isoformat(),
                    "company": self.config.company,
                    "file_type": file_type,
                    "message_id": message_id,
                    "status": "skipped_already_processed",
                })
                continue
            try:
                message = self.service.users().messages().get(userId="me", id=message_id).execute()
                payload = message.get("payload", {})
                parts = self._collect_parts(payload)
                attachment_parts = [p for p in parts if self._is_desired_attachment(p)]
                self._log({
                    "ts": datetime.now().isoformat(),
                    "company": self.config.company,
                    "file_type": file_type,
                    "message_id": message_id,
                    "status": "message_analyzed",
                    "total_parts": len(parts),
                    "attachment_parts": len(attachment_parts),
                    "attachment_filenames": [p.get("filename", "") for p in attachment_parts],
                })
                found = False
                for part in parts:
                    if self._is_desired_attachment(part):
                        part_filename = part.get("filename", "")
                        self._log({
                            "ts": datetime.now().isoformat(),
                            "company": self.config.company,
                            "file_type": file_type,
                            "message_id": message_id,
                            "status": "attempting_attachment_download",
                            "attachment_filename": part_filename,
                        })
                        filename = self._download_attachment(message_id, part)
                        if filename:
                            downloaded_files.append(filename)
                            found = True
                            self._log({
                                "ts": datetime.now().isoformat(),
                                "company": self.config.company,
                                "file_type": file_type,
                                "message_id": message_id,
                                "status": "downloaded_attachment",
                                "filename": filename,
                                "original_filename": part_filename,
                            })
                            break
                if not found:
                    html_bodies = self._extract_html_bodies(payload)
                    self._log({
                        "ts": datetime.now().isoformat(),
                        "company": self.config.company,
                        "file_type": file_type,
                        "message_id": message_id,
                        "status": "checking_html_links",
                        "html_bodies_found": len(html_bodies),
                    })
                    for html in html_bodies:
                        all_links = self._find_all_links(html)
                        matching_links = self._find_links(html)
                        self._log({
                            "ts": datetime.now().isoformat(),
                            "company": self.config.company,
                            "file_type": file_type,
                            "message_id": message_id,
                            "status": "links_analysis",
                            "all_links_count": len(all_links),
                            "matching_links_count": len(matching_links),
                            "all_links": all_links[:10],
                            "matching_links": matching_links[:5],
                            "link_prefixes_configured": self.config.gmail.link_prefixes,
                        })
                        links = matching_links
                        for link in links:
                            self._log({
                                "ts": datetime.now().isoformat(),
                                "company": self.config.company,
                                "file_type": file_type,
                                "message_id": message_id,
                                "status": "attempting_link_download",
                                "link": link,
                            })
                            filename = self._download_from_link(file_type, link)
                            if filename:
                                downloaded_files.append(filename)
                                found = True
                                self._log({
                                    "ts": datetime.now().isoformat(),
                                    "company": self.config.company,
                                    "file_type": file_type,
                                    "message_id": message_id,
                                    "status": "downloaded_link",
                                    "filename": filename,
                                    "link": link,
                                })
                                break
                            else:
                                self._log({
                                    "ts": datetime.now().isoformat(),
                                    "company": self.config.company,
                                    "file_type": file_type,
                                    "message_id": message_id,
                                    "status": "link_download_failed",
                                    "link": link,
                                })
                        if found:
                            break
                
                # Only mark as processed if a file was actually downloaded
                if found:
                    processed_ids.add(message_id)
                    self._log({
                        "ts": datetime.now().isoformat(),
                        "company": self.config.company,
                        "file_type": file_type,
                        "message_id": message_id,
                        "status": "marked_as_processed",
                        "files_downloaded": len(downloaded_files),
                    })
                else:
                    # Don't mark as processed if no file was found
                    # This allows retrying the message later
                    self._log({
                        "ts": datetime.now().isoformat(),
                        "company": self.config.company,
                        "file_type": file_type,
                        "message_id": message_id,
                        "status": "not_marked_as_processed",
                        "reason": "no_file_downloaded",
                    })
                
                # Only mark as read if mark_as_read is enabled
                # This prevents skipping unread messages that failed to download
                if self.config.mark_as_read and found:
                    self._mark_read(message_id)
                
                if not found:
                    links_count = 0
                    if html_bodies:
                        for html in html_bodies:
                            links_count += len(self._find_links(html))
                    self._log({
                        "ts": datetime.now().isoformat(),
                        "company": self.config.company,
                        "file_type": file_type,
                        "message_id": message_id,
                        "status": "no_file_found",
                        "attachment_parts_count": len(attachment_parts),
                        "html_bodies_count": len(html_bodies),
                        "links_found": links_count,
                    })
            except Exception as e:
                self._log({
                    "ts": datetime.now().isoformat(),
                    "company": self.config.company,
                    "file_type": file_type,
                    "message_id": message_id,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
        self._save_processed_ids(list(processed_ids))
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "file_type": file_type,
            "status": "download_complete",
            "files_downloaded": len(downloaded_files),
            "downloaded_files": downloaded_files,
        })
        return downloaded_files

    def run(self, types: List[str]) -> Dict[str, List[str]]:
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "status": "run_started",
            "file_types_requested": types,
        })
        results = {}
        for t in types:
            results[t] = self.download_for_type(t)
        self._log({
            "ts": datetime.now().isoformat(),
            "company": self.config.company,
            "status": "run_complete",
            "results_summary": {k: len(v) for k, v in results.items()},
        })
        return results

    def mark_all_as_read(self, types: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in types:
            ids = self.find_unread_message_ids(t)
            count = 0
            for mid in ids:
                try:
                    self.service.users().messages().modify(userId="me", id=mid, body={"removeLabelIds": ["UNREAD"]}).execute()
                    count += 1
                except Exception as e:
                    self._log({
                        "ts": datetime.now().isoformat(),
                        "company": self.config.company,
                        "file_type": t,
                        "message_id": mid,
                        "status": "error_mark_read",
                        "error": str(e),
                    })
            counts[t] = count
        return counts

    def archive_existing_files(self) -> None:
        base_dir = self.data_dir
        prepared_dir = base_dir / "prepared_archive"
        audit_dir = base_dir / "audit_archive"
        downloads_dir = base_dir / "downloads_archive"
        for d in (prepared_dir, audit_dir, downloads_dir):
            d.mkdir(parents=True, exist_ok=True)
        for ext in ("*.xlsx", "*.csv", "*.parquet"):
            for f in base_dir.glob(ext):
                fname = f.name.lower()
                if "prepared" in fname:
                    dest = prepared_dir / f.name
                elif "audit" in fname:
                    dest = audit_dir / f.name
                else:
                    dest = downloads_dir / f.name
                f.rename(dest)


