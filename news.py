from sympy import div
import yfinance as yf
import bs4 as bs
import requests
import numpy as np


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
        if div:  # Only proceed if div exists
            article = [p.text for p in div.find_all('p')]
            return article
        else:
            return []  # Return empty list if div not found
    except Exception as e:
        print(f"An error occurred while scraping the article: {e}")
        return []

def parse_articles(news):
    articles = []
    
    for item in news:
        try:
            item = item['content']
            
            # Check if URL exists - skip article if not
            try:
                url = item['clickThroughUrl']['url']
            except (KeyError, TypeError):
                continue  # Skip this article if no URL
            
            article_dict = {}
            
            # Safe extraction with np.nan as default for missing values
            try:
                article_dict['title'] = item['title']
            except (KeyError, TypeError):
                article_dict['title'] = np.nan
            
            try:
                article_dict['summary'] = item['summary']
            except (KeyError, TypeError):
                article_dict['summary'] = np.nan
            
            try:
                article_dict['date'] = item['pubDate']
            except (KeyError, TypeError):
                article_dict['date'] = np.nan
            
            article_dict['url'] = url
            
            try:
                article_dict['article'] = scrape_article(url)
            except Exception:
                article_dict['article'] = np.nan
            
            articles.append(article_dict)
            
        except Exception:
            # Skip this entire item if there's any other unexpected error
            continue

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
        import ipdb;ipdb.set_trace()
        return []

if __name__ == "__main__":
    try:
        ticker = "TTD"
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