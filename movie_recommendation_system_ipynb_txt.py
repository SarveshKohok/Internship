import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#data collection AND PRE-PROCESSING

movies_data = pd.read_csv('movies.csv')

movies_data.head()

movies_data.shape

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)

"""replacing a missing value or null value"""

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

"""Combining all the 5 selected features"""

combined_features=movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

print(combined_features)

"""Converting the test data to feature vectors"""

vectorizer=TfidfVectorizer()

feature_vectors=vectorizer.fit_transform(combined_features)

print(feature_vectors)

"""Cosine Similarity
getting similarity scores using cosine similarity

"""

similarity=cosine_similarity(feature_vectors)

print(similarity)

print(similarity.shape)

"""Getting the movie name from the user

"""

movie_name=input('Enter your favourite movie name: ')

"""Creating a list with all the movie names given in the dataset

"""

list_of_all_titles=movies_data['title'].tolist()
print(list_of_all_titles)

"""Finding the close match for the movie name given by the user

"""

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)

close_match=find_close_match[0]
print(close_match)

"""Finding the index of the movie with title"""

index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
print(index_of_the_movie)

"""Getting a list of similar movies"""

similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

"""Sorted the movies based on their similarity score"""

sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
print(sorted_similar_movies)

"""Print the name of similar movies based on the index"""

print('Movies suggested for you: \n')

i=1

for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=movies_data[movies_data.index==index]['title'].values[0]
  if(i<30):
    print(i,'.',title_from_index)
    i+=1

"""**MOVIE RECOMMENDATION SYSTEM**"""

movie_name=input('Enter your favourite movie name: ')


list_of_all_titles=movies_data['title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)

close_match=find_close_match[0]

index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]


similarity_score=list(enumerate(similarity[index_of_the_movie]))



sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

print('Movies suggested for you: \n')

i=1

for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=movies_data[movies_data.index==index]['title'].values[0]
  if(i<30):
    print(i,'.',title_from_index)
    i+=1