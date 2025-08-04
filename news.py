from sympy import div
import yfinance as yf
import bs4 as bs
import requests


def get_yahoo_news(ticker):
    dat = yf.Ticker(ticker)
    news = dat.news
    return news

def scrape_article(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537'}
    response = requests.get(url, headers=headers)
    soup = bs.BeautifulSoup(response.content, 'html.parser')
    try:
        div = soup.find('div', {'class': 'atoms-wrapper'})
        article = [p.text for p in div.find_all('p')]
        return article
    except Exception as e:
        print(f"An error occurred while scraping the article: {e}")
        return []

def parse_articles(news):
    articles = []
    
    for item in news:
        item = item['content']
        
        article_dict = {}
        article_dict['title'] = item['title']
        article_dict['summary'] = item['summary']
        article_dict['date'] = item['pubDate']
        article_dict['url'] = item['clickThroughUrl']['url']
        article_dict['article'] = scrape_article(article_dict['url'])
        
        articles.append(article_dict)

    return articles

def get_extra_info(ticker):
    """
    Fetches additional information about the ticker from Yahoo Finance.
    This includes news articles, summaries, and other relevant data. It then scrapes the full article content from yahoo.
    Args:
        ticker (str): Stock ticker symbol
    Returns:
        list: List of articles with title, summary, date, URL, and full article content
        """
    try:
        news = get_yahoo_news(ticker)
        if not news:
            print(f"No news found for {ticker}")
            return []
        
        articles = parse_articles(news)
        return articles
    except Exception as e:
        print(f"An error occurred while fetching news for {ticker}: {e}")
        return []

if __name__ == "__main__":
    try:
        ticker = "TTE"
        news = get_yahoo_news(ticker)
        
        if news:
            articles = parse_articles(news)

    except Exception as e:
        import traceback
        import ipdb
        
        print(f"An exception occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Drop into ipdb debugger at the point of exception
        ipdb.post_mortem()