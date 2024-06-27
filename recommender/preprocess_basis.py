import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def create_basic_tables(movies_path,ratings_path,tags_path):
    movies = pd.read_csv(movies_path)
    movies['title'] = movies['title'].str.replace(')','',regex=True)
    movies['title'] = movies['title'].str.replace('(','-',regex=True)
    movies['year'] = movies['title'].apply(lambda x: x.split('-')[-1])
    movies['title'] = movies['title'].apply(lambda x: x.split('-')[0])
    movies.loc[~movies['year'].str.isnumeric(),'year'] = np.nan

    ratings = pd.read_csv(ratings_path)
    ratings_by_users = ratings.pivot_table(values='rating',index='userId',columns='movieId')
    ratings_by_users = ratings_by_users.fillna(0)
    ratings_by_users.to_csv('database/ratings_by_users.csv')

    # Movies_feat
    vect = CountVectorizer()
    movies_enhanced = pd.DataFrame(vect.fit_transform(movies.genres).toarray(),columns=vect.get_feature_names_out())
    movies_enhanced = pd.concat([movies,movies_enhanced],axis=1)
    movies_enhanced = movies_enhanced.drop(columns=['title','genres'])
    movies_enhanced = movies_enhanced.set_index('movieId')
    movies_enhanced.to_csv('database/movies_enhanced.csv')

    tags = pd.read_csv(tags_path)
    def clean_text(text):
        res = text.split()
        stop_words = set(stopwords.words('english')) # you can also choose other languages
        #stopwords_removed = [w for w in res if w in stop_words]
        res = [w for w in res if not w in stop_words]
        res = [w.lower() for w in res]
        lemmatizer = WordNetLemmatizer()
        res = [lemmatizer.lemmatize(w) for w in res]
        res = ' '.join(res)
        return res

    tags_enhanced = tags.copy()
    tags_enhanced['tag_clean'] = tags_enhanced['tag'].apply(clean_text)

    vect = CountVectorizer()
    tags_temp = pd.DataFrame(vect.fit_transform(tags_enhanced['tag_clean']).toarray(),columns=vect.get_feature_names_out())

    tags_enhanced = pd.concat([tags_enhanced,tags_temp],axis=1)
    tags_enhanced = tags_enhanced.drop(columns=['tag','tag_clean'])
