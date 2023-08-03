import numpy as np
import pandas as pd

def recommend(username,n):
    movies_enhanced = pd.read_csv('database/movies_enhanced.csv')
    ratings_by_users = pd.read_csv('database/ratings_by_users.csv')

    movies_enhanced_norm = movies_enhanced.copy()
    movies_enhanced_norm['norm'] = movies_enhanced_norm.apply(lambda x : np.sqrt(np.square(x).sum()),axis=1)
    movies_enhanced_norm['norm'] = movies_enhanced_norm['norm'].apply(lambda x : 1 if x==0 else x)
    movies_enhanced_norm = movies_enhanced_norm.apply(lambda x : x/x['norm'],axis=1)
    movies_enhanced_norm = movies_enhanced_norm.drop(columns='norm')

    def get_user_profile(user_id):
        # Computing mean vector
        rated_movies = ratings_by_users.loc[user_id,:]

        weighted_movies_enhanced = movies_enhanced.copy()
        weighted_movies_enhanced = weighted_movies_enhanced.merge(ratings_by_users.T[[1]], left_index=True,right_index=True)
        weighted_movies_enhanced = weighted_movies_enhanced.drop(columns=[1])
        weighted_movies_enhanced = weighted_movies_enhanced.mul(rated_movies,axis=0)

        user_profile = weighted_movies_enhanced.sum(axis=0)
        norm = np.sqrt(np.square(user_profile).sum())
        if norm == 0:
            return 0
        else:
            user_profile = user_profile/norm
        return user_profile

    def get_distances(user_profile):
        movies_temp = movies_enhanced_norm.copy()
        movies_temp = movies_temp.apply(lambda x : x-user_profile,axis=1)
        distances = movies_temp.apply(lambda x : np.sqrt(np.square(x).sum()),axis=1)
        return distances

    def get_top_k(k,user_profile,dist):
        unseen = ratings_by_users.loc[1]
        unseen = unseen[unseen== 0].index
        #dist = get_distances(user_profile)
        dist = dist[unseen].sort_values()
        return dist[:k]

    user_pro = get_user_profile(username)
    distances = get_distances(user_pro)
    top_k = get_top_k(n,user_pro,distances)
    return top_k
