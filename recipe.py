# pylint: disable=missing-docstring, line-too-long, missing-timeout
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Literal
from langchain.agents import tool


SEARCH_URL = "https://recipes.lewagon.com/"


def parse(html):
    ''' return a list of dict {name, difficulty, prep_time} '''
    soup = BeautifulSoup(html, "html.parser")
    return [parse_recipe(article) for article in soup.find_all('div', class_= 'recipe')]


def parse_recipe(article):
    ''' return a dict {name, difficulty, prep_time, id} modeling a recipe'''
    name = article.find('p', class_= 'recipe-name').string.strip()
    difficulty = article.find('span', class_= 'recipe-difficulty').string.strip()
    prep_time = article.find('span', class_= 'recipe-cooktime').string.strip()
    id = article.find('a')['data-id']
    url = SEARCH_URL.strip('/') + article.find('a')['href']

    return {'name': name,
            'difficulty': difficulty,
            'prep_time': prep_time,
            'id': id,
            'url': url
            }


def scrape_from_internet(ingredient, start=1):
    ''' Use `requests` to get the HTML page of search results for given ingredients. '''
    print(f"Scraping page {start}")

    response = requests.get(
        SEARCH_URL,
        params={'search[query]': ingredient, 'page': start}
        )

    # We can check the response history to see if there is a re-direct
    if response.history:
        return None

    return response.text


@tool(parse_docstring=True)
def get_recipes(ingredient: str,
                max_pages: int = 3,
                difficulty_levels: list[Literal["Very Easy", "Easy", "Moderate", "Hard", "Very Hard"]] = [],
                max_prep_time: int = 0,
                min_prep_time: int = 0) -> pd.DataFrame:
    """"Scrape recipes from the internet for a given ingredient.

    This function scrapes recipes from the internet for a given ingredient.
    It returns a DataFrame containing the recipes, including their names,
    difficulty levels, and preparation times. The function allows filtering
    by difficulty levels and preparation times.

    Args:
        ingredient: The ingredient to search for.
        max_pages: The number of pages to scrape. Default is 3.
        difficulty_levels: The difficulty levels to filter by.
            Accepted values are "Very Easy", "Easy", "Moderate", "Hard", and "Very Hard".
            Default is an empty list, which means no filtering.
        max_prep_time: The maximum preparation time in minutes. Default is 0.
            If set to 0, no filtering will be applied.
        min_prep_time: The minimum preparation time in minutes. Default is 0.
            If set to 0, no filtering will be applied.

    Returns:
        A DataFrame containing the recipes for the given ingredient.
    """
    recipes = []

    for page in range(max_pages):
        response = scrape_from_internet(ingredient, page+1)

        if response:
            recipes += parse(response)
        else:
            break

    # Convert prep_time to int
    for recipe in recipes:
        recipe['prep_time'] = int(recipe['prep_time'].split()[0])

    recipes_df = pd.DataFrame(recipes)

    if difficulty_levels:
        recipes_df = recipes_df[recipes_df['difficulty'].isin(difficulty_levels)]

    if max_prep_time:
        recipes_df = recipes_df[recipes_df['prep_time'] <= max_prep_time]

    if min_prep_time:
        recipes_df = recipes_df[recipes_df['prep_time'] >= min_prep_time]

    return recipes_df
