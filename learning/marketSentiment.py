import requests
import os
import pandas as pd


def marketSentiment(df):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("No API key found for News API. Please set the 'NEWS_API_KEY' environment variable.")
        return df

    endpoint = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'bitcoin',
        'sortBy': 'publishedAt',
        'apiKey': api_key,
        'pageSize': 100,
    }
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        print(
            f"Failed to retrieve news data. Status code: {response.status_code}")
        return df

    articles = response.json()['articles']
    if not articles:
        print("No news articles found.")
        return df

    sentiment_scores = []
    for article in articles:
        text = article['title'] + ' ' + article['description']
        response = requests.post(
            'http://text-processing.com/api/sentiment/', data={'text': text})
        sentiment = response.json()['label']
        sentiment_scores.append(sentiment)

    df['sentiment'] = pd.Series(
        sentiment_scores, index=df.index[:len(sentiment_scores)])
    return df
