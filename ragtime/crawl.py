"""
This module defines the logic used to crawl a website and
parse its content for the RAG.
"""

import logging

from ragtime.lib.scraping.crawler import Crawler, ScrapedContent
from ragtime.lib.scraping.processors import Processor


logger = logging.getLogger(__name__)


def printer(content: ScrapedContent):
    if content is None:
        return
    url, link_count = content.get_link_count()
    logger.info(f"\n{url=}\n{link_count=}\n")


processor = Processor(printer, workers=3)


def main():
    url = "https://www.astronomer.io/docs/"
    c = Crawler(workers=4, max_depth=2)
    c.crawl(url=url, processor=processor)


if __name__ == "__main__":
    main()
