# Gmail Customer_Events Download Audit

## Summary

The Customer_Events file download fails because the file is embedded in the email in a non-standard way—requiring a 3-dot menu click to view and download. The current logic expects either (1) traditional MIME attachments or (2) direct `<a href>` links matching a fixed prefix. Embedded/Drive-style content does not match either.

---

## Current Download Flow

### Entry Points

| Location | What Runs |
|----------|-----------|
| `apps/pages/1_Data_Pipeline.py` | Streamlit Download tab → `GmailDownloader(cfg).run(selected_types)` |
| `datahound/scheduler/executor.py` | Scheduled task → `GmailDownloader(cfg).run(config.file_types)` |

### Configuration (McCullough)

From `companies/McCullough Heating and Air/config.json`:

```json
"query_by_type": {
  "Customer_Events": "subject:\"Customer_Events\" is:unread"
},
"link_prefixes": ["https://go.servicetitan.com/PublicResource/File/"]
```

- **Search:** Unread emails with subject containing "Customer_Events"
- **Link filter:** Only links starting with `https://go.servicetitan.com/PublicResource/File/` are used

---

## How the Gmail Downloader Works

**File:** `datahound/download/gmail.py`

### 1. Message Search

- Uses `query_by_type` for the file type (e.g. `Customer_Events`)
- If no unread matches, falls back to read messages for retries
- Fetches full message via `users.messages.get`

### 2. Two Download Paths (in order)

#### Path A: MIME Attachments

- Walks all MIME parts via `_collect_parts`
- Accepts parts that:
  - Have a `filename`
  - Have `attachmentId` in `body`
  - End with `.xlsx`, `.xls`, or `.csv`
- Uses `messages.attachments.get` to download base64 data and save to disk

#### Path B: HTML Link Fallback

- If no valid attachment is found:
  - Extracts HTML bodies from MIME
  - Parses with BeautifulSoup
  - Finds links via `_find_links` (which filters by `link_prefixes`)
  - Fetches each link with `requests.get`
  - Saves response content to a file

### 3. HTML Link Discovery

Current logic looks for:

| Element | How | Prefix filter |
|---------|-----|----------------|
| `<a href="...">` | Direct href | Yes – must match `link_prefixes` |
| `<button onclick="...">` | Regex for URLs in onclick | Yes |
| `<button data-*="...">` | data-url, data-href | Yes |
| `<img>` inside `<a>` | Parent `<a href>` | Yes |
| `<iframe src="...">` | Via `_find_all_links` only | No – not used for download |
| `<meta http-equiv="refresh">` | Via `_find_all_links` only | No – not used for download |

`_find_links` returns only links that start with `link_prefixes` (`https://go.servicetitan.com/PublicResource/File/`). `_find_all_links` is used for logging.

---

## Why Customer_Events Fails (3-Dot Embed)

### Likely Structure

If the file shows up only after opening a 3-dot menu, it is likely:

1. **Drive-style embed** – Link/file hosted on Google Drive or similar, embedded in the email
2. **Progressive disclosure UI** – Link is inside a collapsible/expandable section or behind a menu
3. **Different URL format** – Real download URL might use:
   - `https://drive.google.com/...`
   - `https://docs.google.com/...`
   - A different ServiceTitan path
   - Or a redirect that differs from `go.servicetitan.com/PublicResource/File/`

### Gaps in Current Logic

| Issue | Cause | Consequence |
|-------|-------|-------------|
| Not a standard MIME attachment | File is referenced via link, not `attachmentId` | Path A never runs |
| Link prefix is too narrow | Only `go.servicetitan.com/PublicResource/File/` | Path B ignores other URLs |
| Link inside Drive/iframe | URL may live in iframe `src` or similar | `_find_links` doesn’t use iframe links |
| Hidden/collapsible markup | Link in a collapsed section or behind JavaScript | Parser finds it if it’s in static HTML, but not if it’s injected by JS |
| 3-dot menu loads content dynamically | Menu action triggers new page or API call | Raw email HTML has no direct link to the file |

---

## Logging and Debugging

Logs go to `{pipeline_dir}/download.jsonl` (via `central_logging.config.pipeline_dir`).

Relevant log events:

- `searching_messages`, `search_complete` – Query and hit count
- `message_analyzed` – MIME parts and attachment counts
- `checking_html_links`, `links_analysis` – HTML bodies and link counts
- `all_links`, `matching_links` – Which links were found vs used
- `no_file_found` – No attachment and no usable link
- `link_download_failed` – Link found but HTTP GET failed

### How to Debug

1. Run a Customer_Events download and inspect `download.jsonl`
2. In `links_analysis`:
   - Compare `all_links_count` vs `matching_links_count`
   - If `all_links` has useful URLs but `matching_links` is empty, the prefix filter is too strict
3. Manually open a Customer_Events email:
   - Open the 3-dot menu and try to download
   - Check the actual download URL (browser dev tools / “Copy link”)

---

## Recommended Fixes

### 1. Inspect the Real Download URL

- Open a Customer_Events email, use the 3-dot menu to download
- Capture the full URL and see if it matches current `link_prefixes`
- Add new prefixes if ServiceTitan changed their URLs

### 2. Broaden Link Matching for Customer_Events

- For Customer_Events, optionally allow:
  - `https://drive.google.com/...`
  - `https://docs.google.com/...`
  - Or a per–file-type override for link prefixes

### 3. Use Iframe/Additional Sources

- Extend `_find_links` to consider `<iframe src>` when `src` is a file or export URL
- Check `data-*` on divs/spans (e.g. `data-download-url`, `data-file-url`)
- Check `src` on `<object>` and `<embed>`

### 4. Handle Google Drive Links

- If the real link is `drive.google.com/file/d/FILE_ID/view`
- Convert to direct download: `drive.google.com/uc?export=download&id=FILE_ID`
- Use Drive API or authenticated `requests` if the file is not publicly shared

### 5. Fallback: Manual or Browser-Based Flow

- If content is only available through interactive UI:
  - Add a manual “paste URL” option in the app
  - Or run a Selenium/Playwright flow that opens the email and mimics the 3-dot click to obtain the URL or file

---

## File Type Chain

| Stage | File type key | Mapping |
|-------|----------------|---------|
| Gmail search | `Customer_Events` | Subject "Customer_Events" |
| Download | `Customer_Events` | Same key |
| Prepare | `customer_events` | Maps to "customers" in `pick_newest_by_type` |
| Upsert | `customers` | `file_type_to_master`: Customers.xlsx |

So Customer_Events emails must yield a file whose name contains `"customer_events"` (or similar) so the prepare step can match it and produce the Customers master file.
