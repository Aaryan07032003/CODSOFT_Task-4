import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('Moviess.csv')

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
data['combined_features'] = data['Title'] + " " + data['Genre'] + " " + data['Cast'] + " " + data['Description']
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Cosine similarity 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Movie recommendations 
def get_movie_recommendations(movie_title, cosine_sim, data, n=5):
    if movie_title in data['Title'].values:
        idx = data[data['Title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1] 
        movie_indices = [i[0] for i in sim_scores]
        recommendations = data['Title'].iloc[movie_indices].unique()  
        return recommendations
    else:
        return "Movie not found in the dataset"

while True:
    user_movie_title = input("Enter a movie title (type 'stop' to end): ")
    
    if user_movie_title.lower() == 'stop':
        break
    
    recommendations = get_movie_recommendations(user_movie_title, cosine_sim, data)
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("Recommendations for", user_movie_title, ":")
        for recommendation in recommendations:
            print(recommendation)
