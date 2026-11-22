"""
Web URL parser using Crawl4AI — extracts clean Markdown from web pages.
Used for both initial ingestion and scheduled refreshes.
"""

from pathlib import Path
import re
from typing import Optional

from loguru import logger

from ingestion.parsers.base import BaseDocumentParser, CanonicalDocument, SourceType


class WebParser(BaseDocumentParser):
    supported_mime_types = ["text/html", "text/plain"]
    supported_extensions = [".html", ".htm"]

    async def parse(
        self,
        source_id: str,
        tenant_id: str,
        file_path: Optional[Path] = None,
        url: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> CanonicalDocument:
        if url is None:
            raise ValueError("WebParser requires a url")

        logger.info(f"Crawling URL: {url}")

        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

            browser_cfg = BrowserConfig(headless=True, verbose=False)
            run_cfg = CrawlerRunConfig(
                word_count_threshold=10,
                remove_overlay_elements=True,
                process_iframes=False,
            )
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)
                markdown_content = result.markdown or result.cleaned_html or ""
                page_title = result.metadata.get("title", url) if result.metadata else url

        except Exception as e:
            logger.warning(f"Crawl4AI unavailable ({e}) — falling back to httpx + BeautifulSoup")
            markdown_content, page_title = await self._fallback_web_parse(url)

        if not markdown_content or not markdown_content.strip():
            raise RuntimeError(f"Web extraction produced no readable content for {url}")

        # Ensure page content has a root Markdown heading
        cleaned_md = markdown_content.strip()
        if page_title and not cleaned_md.startswith("#"):
            markdown_content = f"# {page_title}\n\n{cleaned_md}"

        return CanonicalDocument(
            source_id=source_id,
            tenant_id=tenant_id,
            source_type=SourceType.WEB,
            source_name=page_title,
            source_uri=url,
            markdown_content=markdown_content,
            page_texts=[markdown_content],
            total_pages=1,
            extra_metadata={**(extra_metadata or {}), "url": url},
        )

    async def _fallback_web_parse(self, url: str) -> tuple[str, str]:
        try:
            import httpx
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else url

                # Remove non-content tags
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "title"]):
                    tag.decompose()

                # Remove boilerplate header tickers, score strips, ads, quick links
                for selector in [
                    ".ci-hsb-container",
                    "[class*='ticker']",
                    "[class*='scoreboard-strip']",
                    "[class*='ad-']",
                    "[class*='quick-link']",
                    "[class*='global-nav']",
                ]:
                    for el in soup.select(selector):
                        el.decompose()

                # Convert HTML headings (h1-h6) to Markdown headings before extracting text
                for level in range(1, 7):
                    tag_name = f"h{level}"
                    prefix = "#" * min(level, 3) + " "
                    for h in soup.find_all(tag_name):
                        h_text = h.get_text(strip=True)
                        if h_text:
                            h.replace_with(f"\n\n{prefix}{h_text}\n\n")
                        else:
                            h.decompose()

                # Convert HTML tables to Markdown tables
                for table in soup.find_all("table"):
                    table_lines = []
                    rows = table.find_all("tr")
                    for i, row in enumerate(rows):
                        cells = row.find_all(["th", "td"])
                        row_cells = [c.get_text(strip=True).replace("\n", " ") for c in cells]
                        if not any(row_cells):
                            continue
                        row_str = "| " + " | ".join(row_cells) + " |"
                        table_lines.append(row_str)
                        if i == 0:
                            sep = "| " + " | ".join(["---"] * len(row_cells)) + " |"
                            table_lines.append(sep)
                    if table_lines:
                        table.replace_with("\n\n" + "\n".join(table_lines) + "\n\n")

                text = soup.get_text(separator="\n\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                if title and not text.strip().startswith("#"):
                    text = f"# {title}\n\n{text}"
                return text, str(title)
        except Exception as e:
            logger.error(f"Fallback web parse failed for {url}: {e}")
            return "", url
