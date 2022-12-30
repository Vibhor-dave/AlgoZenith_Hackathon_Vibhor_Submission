from cmath import nan

import pandas as pd
import numpy as np
from ast import literal_eval  # evaluate strings containing Python code in the current Python environment
from nltk.stem.snowball import SnowballStemmer  # Removing stem words
from sklearn.feature_extraction.text import CountVectorizer  # To convert text to numerical data
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings  # disable python warnings

warnings.filterwarnings("ignore")  # ignoring warnings

# reading data
movies_data = pd.read_csv("C:\\Users\\vibho\\anaconda3\\envs\\mlenv\\movies_metadata.csv", low_memory=False)
credits = pd.read_csv('C:\\Users\\vibho\\anaconda3\\envs\\mlenv\\credits.csv')
keywords = pd.read_csv('C:\\Users\\vibho\\anaconda3\\envs\\mlenv\\keywords.csv')
links_small = pd.read_csv('C:\\Users\\vibho\\anaconda3\\envs\\mlenv\\links_small.csv')
ratings = pd.read_csv("C:\\Users\\vibho\\anaconda3\\envs\\mlenv\\ratings_small.csv")

# removing vote_average and vote_count
movies_data = movies_data.dropna(subset=['vote_average', 'vote_count'])


def weighted_rating(v, R, C, m):
    '''

    This function calculate weighted rating of a movies using IMDB formula

    Parameters: v (int): vote count
                R (int): vote average
    Returns: (float) IMDB score

    '''
    return ((v / (v + m)) * R) + ((m / (m + v)) * C)


C = movies_data['vote_average'].mean()  # mean vote across all data
m = movies_data['vote_count'].quantile(0.95)  # movies with more than 95% votes is taken (95 percentile)

# Taking movies whose vote count is greater than m
copy_df = movies_data.copy()
top_movies = copy_df.loc[movies_data['vote_count'] >= m]
top_movies = top_movies.reset_index()

top_movies['score'] = ''

for i in range(top_movies.shape[0]):
    v = top_movies['vote_count'][i]  # number of vote count of the movie
    R = top_movies['vote_average'][i]  # average rating of the movie
    top_movies['score'][i] = weighted_rating(v, R, C, m)

top_movies = top_movies.sort_values('score',
                                    ascending=False)  # sorting movies in descending order according to score
top_movies = top_movies.reset_index()

# top_movies[['title', 'vote_count', 'vote_average', 'score']].head(20) # top 20 movies
t1 = top_movies[['title', 'score']].head(5)


# finding the best overall
def getTopRec():
    return t1


# creating a function to return genere based top movies
def create_gen_based(name):
    # finding generes set
    genres = set()
    top_movies_copy = top_movies.copy()  # using copy so that top_movies can be reused
    top_movies_copy['genres'] = top_movies_copy['genres'].apply(literal_eval)
    for i in range(top_movies_copy['genres'].shape[0]):  # converting string in map
        for x in top_movies_copy['genres'][i]:
            genres.add(x['name'])

    # creating map of string (genre name) and movies names(dataframe)
    genres_based = dict()
    for i in range(top_movies_copy['genres'].shape[0]):
        for x in top_movies_copy['genres'][i]:
            if x['name'] not in genres_based.keys():
                genres_based[x['name']] = pd.DataFrame(columns=top_movies.columns)
            genres_based[x['name']] = genres_based[x['name']].append(top_movies_copy.iloc[i])

    recommended = genres_based_rcmnd(name, genres, genres_based)
    return recommended  # returning recommendation on the basis of generes


def genres_based_rcmnd(name, genres, genres_based):
    if name not in genres:
        return None
    else:
        return genres_based[name]['title'].head(10)


def create_language_based(name):
    # finding language based
    language = set()
    for i in range(top_movies['original_language'].shape[0]):  # converting string in map
        language.add(top_movies['original_language'][i])

    # creating map of string (genre name) and movies names(dataframe)
    language_based = dict()
    for i in range(top_movies['original_language'].shape[0]):
        if top_movies['original_language'][i] not in language_based.keys():
            language_based[top_movies['original_language'][i]] = pd.DataFrame(columns=top_movies.columns)
        language_based[top_movies['original_language'][i]] = language_based[top_movies['original_language'][i]].append(
            top_movies.iloc[i])

    recommended = language_based_rcmnd(name, language, language_based)
    return recommended  # returning language based recommended list


def language_based_rcmnd(name, language, language_based):
    if name not in language:
        return None
    else:
        return language_based[name]['title'].head(10)


# print(create_language_based('zh'))
# print(create_collection_based('Three Colors Collection'))
# print(create_gen_based('Action'))

movies_data['id'] = movies_data['id'].astype('int')

movies_data = movies_data.merge(credits, on='id')
movies_data = movies_data.merge(keywords, on='id')

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype(
    int)  # finding all those present in 'tmdbId'

smd = movies_data[movies_data['id'].isin(links_small)]
smd.reset_index()


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Applying literal_eval to get the right data type from the expression of string
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['genres'] = smd['genres'].apply(literal_eval)

smd['director'] = smd['crew'].apply(get_director)

# Taking all the movie cast in a list and then taking only the top 3 cast
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd['cast'] = smd['cast'].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x])  # Strip Spaces and Convert to Lowercase

smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['genres'] = smd['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['genres'] = smd['genres'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(
    lambda x: [x, x, x])  # giving more weight to the director relative to the entire cast

# Stemming the words
stemmer = SnowballStemmer('english')

smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# combining keywords, cast, director and genres
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
# souping all important things in one column
count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)
# finding cosine similarity

titles = smd['title']
index = list()
for i in range(0, len(smd.index)):
    index.append(i)
indices = pd.Series(index, index=smd['title'])  # Creating a mapping between movie and title and index
indices.reset_index()


def get_recommendations(title):
    idx = indices[title]  # movie id corrosponding to the given title
    sim_scores = list(enumerate(cosine_sim[idx]))  # list of cosine similarity scores value along the given index
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # sorting the given scores in ascending order
    sim_scores = sim_scores[1:31]  # Taking only the top 30 scores
    movie_indices = [i[0] for i in sim_scores]  # Finding the indices of 30 most similar movies
    return titles.iloc[movie_indices]

