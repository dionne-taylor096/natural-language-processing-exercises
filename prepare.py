#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import env
from env import host, user, password
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wrangle
import pandas as pd
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from nltk.probability import FreqDist
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# In[ ]:

import pandas as pd
import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Define helper functions
def basic_clean(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    return text

def tokenize(text):
    return word_tokenize(text)

def stem(text):
    ps = PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    text_stemmed = ' '.join(stems)
    return text_stemmed

def lemmatize(text):
    wnl = WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    text_lemmatized = ' '.join(lemmas)
    return text_lemmatized

def remove_stopwords(text, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words('english')
    if len(extra_words) > 0:
        stopword_list += extra_words
    if len(exclude_words) > 0:
        stopword_list = list(filter(lambda x: x not in exclude_words, stopword_list))
    words = text.split()
    filtered_words = [w for w in words if w not in stopword_list]
    return ' '.join(filtered_words)

# Define your main function
def clean_text_nlp(df, original_cols, new_cols):
    """
    This function will take in a dataframe, the name of the column to clean,
    and a list of new column names to create from the cleaning process.
    It will produce the following columns:

    - clean: to hold the normalized and tokenized original with the stopwords removed.
    - stemmed: to hold the stemmed version of the cleaned data.
    - lemmatized: to hold the lemmatized version of the cleaned data.
    """
    for i, col in enumerate(original_cols):
        df[new_cols[i]] = (df[col]
                           .astype(str)
                           .apply(basic_clean)
                           .apply(tokenize)
                           .apply(' '.join)
                           .apply(remove_stopwords))
        df[new_cols[i]+'_stemmed'] = df[new_cols[i]].apply(stem)
        df[new_cols[i]+'_lemmatized'] = df[new_cols[i]].apply(lemmatize)
    return df


def sent_analyze(df, text_column):
    analyzer = SentimentIntensityAnalyzer()

    df['polarity'] = df[text_column].apply(lambda x: analyzer.polarity_scores(x))

    # Change data structure
    df = pd.concat(
        [df.drop(['polarity'], axis=1), 
         df['polarity'].apply(pd.Series)], axis=1)

    # Create new variable with sentiment "neutral," "positive" and "negative"
    df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')

    # Blog with highest positive sentiment
    print("Blog with highest positive sentiment:")
    print(df.loc[df['compound'].idxmax()].values)

    # Blog with highest negative sentiment 
    print("Blog with highest negative sentiment:")
    print(df.loc[df['compound'].idxmin()].values)

    # Number of tweets 
    sns.countplot(y='sentiment', 
                 data=df, 
                 palette=['#b2d8d8',"#008080", '#db3d13']
                 )

import pandas as pd
import numpy as np

def clean_log(df):

    # Rename the columns to match the data
    df.columns = ['date', 'time', 'page', 'user_id', 'cohort_id', 'ip']

    # Convert date and time to datetime and combine them
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    # Calculate the time spent on each page
    df['time_spent'] = df['datetime'].diff().dt.total_seconds().fillna(0)

    # Replace 'NaN' with 'unknown' in the DataFrame
    df.replace(np.nan, '0', inplace=True)

    # Create a session ID for each user based on a 30-minute session time
    df['session_id'] = (df['datetime'].diff() > pd.Timedelta(minutes=30)).cumsum()

    # Convert the 'cohort_id' column to integer data type
    df['cohort_id'] = df['cohort_id'].astype(int)
    
    print("The clean_log() function takes a pandas DataFrame df as an input and performs several cleaning and preprocessing tasks on it.First, it replaces any NaN values in the DataFrame with 0. Then, it creates a new column 'session_id' by using the diff() method to calculate the time difference between consecutive entries in the 'datetime' column. If the time difference is greater than 30 minutes, the entry is considered to be a new session and a new session ID is assigned. The function then converts the 'cohort_id' column to integer data type, uses str.contains() to filter rows based on the substring 'United States' in the 'location' column, and selects non-US locations using the negation operator ~. Next, the function drops the 'date' and 'time' columns, and categorizes the remaining pages into three categories ('web_development', 'data_science', 'curriculum_infra') based on the presence of specific keywords in the 'page' column. Finally, the function returns the cleaned DataFrame.")
    
    # Return the cleaned dataframe
    return df

import geoip2.database
import pandas as pd
import numpy as np

def lookup_location_info(df):
    reader = geoip2.database.Reader('GeoLite2-City.mmdb')
    locations = []
    for ip in df['ip']:
        try:
            response = reader.city(ip)
            location = response.country.name
            if response.subdivisions.most_specific.name:
                location = response.subdivisions.most_specific.name + ', ' + location
            locations.append(location)
        except geoip2.errors.AddressNotFoundError:
            locations.append('Suspect')
    
    df['location'] = locations
    print("The function lookup_location_info uses the MaxMind GeoIP2 database to extract location information from IP addresses in a given DataFrame. The function reads the location data from the GeoLite2-City.mmdb file and uses it to retrieve the country and, if available, subdivision information for each IP address in the DataFrame. The extracted location information is then added to the DataFrame as a new column called location. If an IP address is not found in the database, the location is labeled as 'Suspect'.")
    return df

import pandas as pd


import random

def random_categories(df):
    # Define fish names for each category
    web_dev_animal_names = ['Lion', 'Tiger', 'Leopard', 'Cheetah', 'Jaguar', 'Panther', 'Cougar', 'Lynx', 'Bobcat', 'Ocelot', 'Caracal', 'Serval', 'Wolf', 'Fox', 'Coyote', 'Jackal', 'Raccoon', 'Skunk', 'Badger', 'Weasel', 'Ferret', 'Mink', 'Otter', 'Hyena', 'Honey Badger', 'Puma', 'Wolverine', 'Gazelle', 'Antelope', 'Elk', 'Moose', 'Reindeer', 'Giraffe', 'Zebra', 'Hippopotamus', 'Rhino', 'Elephant', 'Gorilla', 'Chimpanzee', 'Orangutan', 'Baboon', 'Marmoset', 'Lemur', 'Sloth', 'Kangaroo', 'Koala', 'Wombat', 'Platypus', 'Emu', 'Ostrich']

    data_sci_fish_names = ['Salmon', 'Trout', 'Bass', 'Mullet', 'Carp', 'Goby', 'Pike', 'Tilapia', 'Sturgeon', 'Sardine', 'Swordfish', 'Anchovy', 'Herring', 'Whiting', 'Ling', 'Pilchard', 'Hake', 'John Dory', 'Red Snapper', 'Sea Bass', 'Dolphin Fish', 'Gurnard', 'Turbot', 'Octopus', 'Squid', 'Cuttlefish', 'Crab', 'Lobster', 'Prawn', 'Shrimp', 'Clam', 'Oyster', 'Mussel', 'Cockle', 'Scallop', 'Abalone', 'Periwinkle', 'Barnacle', 'Jellyfish', 'Sea Cucumber', 'Sea Urchin', 'Starfish', 'Seahorse', 'Nautilus', 'Sponge', 'Corals', 'Anemone', 'Sea Slug']

    # Assign random fish names to each cohort_id based on category
    cohort_names = {}
    for i, row in df.iterrows():
        if row['category'] == 'web_development':
            nick_name = random.choice(web_dev_animal_names)
        else:
            nick_name = random.choice(data_sci_fish_names)
        if row['cohort_id'] not in cohort_names:
            cohort_names[row['cohort_id']] = nick_name
        df.loc[i, 'nick_name'] = nick_name
    print(" This function, random_categories, generates random fish or animal names for each cohort_id based on their category. The function takes a DataFrame df as input and assigns random names from two different lists of animal and fish names to each cohort_id based on their category. If the category of the cohort is 'web_development', a random animal name is chosen from web_dev_animal_names list; otherwise, a random fish name is chosen from data_sci_fish_names list. For each cohort_id, the function creates a dictionary cohort_names and assigns a random name as a value for each key representing a cohort_id. Then it iterates through each row of the DataFrame and assigns the corresponding random name from the cohort_names dictionary to each row. Finally, the function returns the modified DataFrame with an additional column called nick_name, which contains the randomly assigned animal or fish name for each cohort_id.")   
    return df



def custom_visual():
    """
    This function configures some visual settings to enhance the readability and aesthetics of data visualizations.

    The settings include configuring the Seaborn style to "darkgrid" for better visual contrast and readability,
    setting the Matplotlib style to "dark_background" for a visually appealing dark theme, setting the default
    float format in Pandas to display two decimal places, setting the maximum column width in Pandas to display the
    entire content without truncation, setting the display width in Pandas to match the terminal/console width, and
    resetting the column header justification in Pandas to its default (left-aligned).

    Additionally, the function sets the maximum number of rows to display to 400.
    """
    # Set the Seaborn style to "darkgrid" for better visual contrast and readability
    sns.set_style("darkgrid")

    # Set the Matplotlib style to "dark_background" for a visually appealing dark theme
    plt.style.use('dark_background')

    # Configure the default float format in Pandas to display two decimal places
    pd.options.display.float_format = '{:20,.2f}'.format

    # Set the maximum column width in Pandas to display the entire content without truncation
    pd.set_option('display.max_colwidth', None)

    # Set the display width in Pandas to match the terminal/console width
    pd.set_option('display.width', None)

    # Reset the column header justification in Pandas to its default (left-aligned)
    pd.reset_option("colheader_justify", 'right')
    
    # Set the maximum number of rows to display to 400
    pd.set_option('display.max_rows', 400)
    
    print("This function configures some visual settings to enhance the readability and aesthetics of data visualizations. The settings include configuring the Seaborn style to darkgrid for better visual contrast and readability, setting the Matplotlib style to dark_background for a visually appealing dark theme, setting the default float format in Pandas to display two decimal places, setting the maximum column width in Pandas to display the entire content without truncation, setting the display width in Pandas to match the terminal/console width, and resetting the column header justification in Pandas to its default (left-aligned). Additionally, the function sets the maximum number of rows to display to 400.")



def productivity_class(value):
    if value < 0.33:
        return 0  # low productivity
    elif value < 0.66:
        return 1  # medium productivity
    else:
        return 2  # high productivity
    

import pandas as pd
import numpy as np

def clean_up(df):

    """
Parameters:
-----------
df : pandas DataFrame
    DataFrame to be cleaned

Returns:
--------
df : pandas DataFrame
    Cleaned DataFrame
    """ 
    # Replace 'NaN' with 'unknown' in the DataFrame
    df.replace(np.nan, '0', inplace=True)
    
    df['session_id'] = (df['datetime'].diff() > pd.Timedelta(minutes=30)).cumsum()
    
    # Convert the 'cohort_id' column to integer data type
    df['cohort_id'] = df['cohort_id'].astype(int)
    
    # Use str.contains() to filter rows based on a substring
    us_locations = df[df['location'].str.contains('United States', na=False)]

    # Use ~ to negate the condition and select non-US locations
    non_us_locations = df[~df['location'].str.contains('United States', na=False)]
    
    # Drop the 'date' and 'time' columns
    df.drop(['date', 'time'], axis=1, inplace=True)
    
    # Define the keywords for each category
    web_dev_keywords = ['java', 'html', 'jdb', 'object', 'jquery']
    data_sci_keywords = ['sql', 'classification', 'python', 'git', 'clustering','anomaly','regression','nlp']
    curriculum_infra_keywords = ['mkdoc','toc', '/', 'appendix','spring','asdf']
    
    # Categorize the pages based on the keywords
    df['category'] = ''
    for i, row in df.iterrows():
        if any(keyword in row['page'] for keyword in web_dev_keywords):
            df.loc[i, 'category'] = 'web_development'
        elif any(keyword in row['page'] for keyword in data_sci_keywords):
            df.loc[i, 'category'] = 'data_science'
        elif any(keyword in row['page'] for keyword in curriculum_infra_keywords):
            df.loc[i, 'category'] = 'curriculum_infra'
    
    # View the DataFrame
    df
    print("The clean_up() function removes any missing values in the DataFrame and replaces them with '0'. It then creates a 'session_id' column based on a 30-minute time difference, converts the 'cohort_id' column to an integer data type, filters the DataFrame based on whether the location is in the United States or not, drops the 'date' and 'time' columns, and categorizes the pages based on a set of predefined keywords. The function returns the cleaned DataFrame.")
    
    return df


import geoip2.database
import pandas as pd
import numpy as np

def lookup_location_info(df):
    reader = geoip2.database.Reader('GeoLite2-City.mmdb')
    locations = []
    for ip in df['ip']:
        try:
            response = reader.city(ip)
            location = response.country.name
            if response.subdivisions.most_specific.name:
                location = response.subdivisions.most_specific.name + ', ' + location
            locations.append(location)
        except geoip2.errors.AddressNotFoundError:
            locations.append('Suspect')
    
    df['location'] = locations
    return df

def process_text_columns(df, text_cols, stopwords_extra=[]):
    """
    This function takes in a DataFrame and a list of text column names to process. 
    It applies a series of text processing steps to the specified columns and returns 
    the cleaned DataFrame along with the top 10 most common words in the 'content' column.
    """
    for col in text_cols:
        df[col] = df[col].astype(str).str.lower()

        regexp = RegexpTokenizer('\w+')
        df[col+'_token'] = df[col].apply(regexp.tokenize)

        stopwords = nltk.corpus.stopwords.words("english")
        stopwords.extend(stopwords_extra)

        df[col+'_token'] = df[col+'_token'].apply(lambda x: [item for item in x if item not in stopwords])
        df[col+'_string'] = df[col+'_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))

        all_words = ' '.join([word for word in df[col+'_string']])
        tokenized_words = nltk.tokenize.word_tokenize(all_words)
        fdist = FreqDist(tokenized_words)

        df[col+'_string_fdist'] = df[col+'_token'].apply(lambda x: ' '.join([item for item in x if fdist[item] >= 1 ]))

        wordnet_lem = WordNetLemmatizer()
        df[col+'_string_lem'] = df[col+'_string_fdist'].apply(wordnet_lem.lemmatize)

        df['is_equal']= (df[col+'_string_fdist']==df[col+'_string_lem'])

        all_words_lem = ' '.join([word for word in df[col+'_string_lem']])
        words = nltk.word_tokenize(all_words_lem)
        fd = FreqDist(words)

        top_10 = fd.most_common(10)
        fdist = pd.Series(dict(top_10))
    
    return df, fdist


""" 

Ask yourself:

   1. If your corpus is 493KB, would you prefer to use stemmed or lemmatized text?
   2. If your corpus is 25MB, would you prefer to use stemmed or lemmatized text?
   3. If your corpus is 200TB of text and you're charged by the megabyte for your hosted computational
    resources, would you prefer to use stemmed or lemmatized text?

1. The choice between stemming and lemmatization can depend on the specific requirements of your project and the trade-off between speed and accuracy.

2. If the increased accuracy and contextual understanding are essential for your project, lemmatization could be the preferred choice, even if it comes with a slightly slower processing time.

3. Considering the cost implications, stemming might be a more practical choice in this scenario.

"""
