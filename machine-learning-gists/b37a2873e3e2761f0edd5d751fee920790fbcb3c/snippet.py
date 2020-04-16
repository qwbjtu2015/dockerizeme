import pandas as pd

# pass in column names for each CSV and read them using pandas.
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,encoding='latin-1')
rating = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

rp = rating.pivot_table('rating', index='movie_id',columns='user_id')
test = rp[1]
sim_toby = rp.corrwith(test)

#print sim_toby
rating_c = rating[test[rating.movie_id].isnull().values & (rating.user_id != 1)]
rating_c['similarity'] = rating_c['user_id'].map(sim_toby.get)
rating_c['sim_rating'] = rating_c.similarity * rating_c.rating

recommendation = rating_c.groupby('movie_id').apply(lambda s: s.sim_rating.sum() / s.similarity.sum())
print recommendation.sort_values(ascending=False)