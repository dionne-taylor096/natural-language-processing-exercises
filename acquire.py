#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import os
import env
import requests
from bs4 import BeautifulSoup
import pandas as pd
from env import host, user, password

# In[ ]:
def get_news_articles():
    base_url = 'https://inshorts.com/en/read/'
    categories = ['business', 'sports', 'technology', 'entertainment']
    headers = {'User-Agent': 'Mozilla/5.0'}
    article_list = []

    for category in categories:
        url = base_url + category
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('div', class_='news-card')
        for article in articles:
            title = article.find('span', itemprop='headline').text.strip()
            content = article.find('div', itemprop='articleBody').text.strip()
            article_dict = {'title': title, 'content': content, 'category': category}
            article_list.append(article_dict)

    return article_list

def fetch_data(url):
    data = []

    while url:
        response = requests.get(url)
        json_data = response.json()
        data.extend(json_data["results"])
        url = json_data["next"]

    return data

def grab_csv_data(api_url, output_file):
    if not os.path.exists(output_file):
        response = requests.get(api_url)

        if response.status_code == 200:
            csv_data = response.text
            with open(output_file, 'w') as f:
                f.write(csv_data)
            print(f"CSV data saved to {output_file}")
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    return pd.read_csv(output_file)
    


def acquire_log_data():
    filename = 'anonymized-curriculum-access.txt'
    print("The acquire_log_data() function reads in data from a file named 'anonymized-curriculum-access.txt' by using the pd.read_csv() method. If the file is found in the current directory, it is read in and a Pandas DataFrame is returned. If the file is not found, the function prints 'The file doesn't exist' and recursively calls itself until the file is found.")

    if os.path.isfile(filename):
        return pd.read_csv(filename, delimiter=' ', header=None)
    else:
        print("The file doesn't exist")
        df = acquire_log_data()
        return df
    
import requests
from bs4 import BeautifulSoup

def get_full_blog_articles():
    headers = {'User-Agent': 'Codeup Data Science'}
    base_url = 'https://codeup.com/blog/page/'
    page_num = 1
    article_list = []

    while True:
        url = base_url + str(page_num)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        if not articles:
            break
        
        for article in articles:
            title_element = article.find('h2', class_='entry-title')
            if title_element is not None:
                title = title_element.text.strip()
                date_element = article.find('p', class_='post-meta')
                date = date_element.text.strip() if date_element is not None else ''
                content_element = article.find('div', class_='post-content')
                read_more = content_element.find('a', class_='more-link')
                content_url = read_more['href']
                content_response = requests.get(content_url, headers=headers)
                content_soup = BeautifulSoup(content_response.text, 'html.parser')
                content_div = content_soup.find('div', class_='entry-content')
                content = content_div.text.strip() if content_div is not None else ''
            
                article_dict = {
                    'title': title,
                    'date': date,
                    'content': content
                }
            
                article_list.append(article_dict)
        
        page_num += 1
    
    return article_list

