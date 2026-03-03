import feedparser

def fetch_news():
    urls = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/edition.rss",
    "https://www.technologyreview.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss"
]

    articles = []

    for url in urls:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            content = (
                getattr(entry, "summary", None) or
                getattr(entry, "description", None) or
                (entry.content[0].value if hasattr(entry, "content") else None) or
                entry.title
            )

            articles.append({
                "title": getattr(entry, "title", ""),
                "content": content,
                "source": getattr(entry, "link", url)
            })

    return articles