from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Load user comments to calculate average rating and sentiment for each movie
def load_user_comments(file_path):
    user_comments = pd.read_csv(file_path)
    movie_stats = user_comments.groupby('qid').agg(
        average_sentiment=('sentiment', 'mean'),
        average_rating=('rating', lambda x: x.mean() / 10)  # Normalize rating to 0-1 scale
    ).to_dict('index')
    return movie_stats

# Function to calculate similarity score between two movies
def calculate_similarity(movie1, movie2, movie_stats):
    # Updated weights, including average rating and sentiment
    weights = {
        "genre": 0.55,  # Further increased weight for genre
        "director": 0.07,  # Reduced weight for director
        "cast": 0.03,  # Reduced weight for cast
        "production_company": 0.1,  # Reduced weight for production company
        "country_of_origin": 0.03,  # Reduced weight for country of origin
        "release_date": 0.03,  # Reduced weight for release date
        "instance_of": 0.1,  # Adjusted weight for instance_of
        "average_rating": 0.045,  # Adjusted weight for average rating
        "average_sentiment": 0.045  # Adjusted weight for average sentiment
    }

    score = 0
    matched_attributes = []

    for attr, weight in weights.items():
        if attr in movie1 and attr in movie2:
            movie1_values = movie1[attr] if isinstance(movie1[attr], list) else [movie1[attr]]
            movie2_values = movie2[attr] if isinstance(movie2[attr], list) else [movie2[attr]]

            movie1_values_normalized = set(map(str.lower, movie1_values))
            movie2_values_normalized = set(map(str.lower, movie2_values))

            common_elements = movie1_values_normalized.intersection(movie2_values_normalized)

            if common_elements:
                overlap_ratio = len(common_elements) / max(len(movie1_values_normalized), len(movie2_values_normalized))
                score += weight * overlap_ratio
                matched_attributes.append(attr)
        elif attr == "release_date":
            date_score = calculate_date_proximity(movie1.get(attr), movie2.get(attr), weight)
            score += date_score
            if date_score > 0:
                matched_attributes.append(attr)
        elif attr in ["average_rating", "average_sentiment"]:
            qid1 = movie1.get("qid")
            qid2 = movie2.get("qid")
            if qid1 in movie_stats and qid2 in movie_stats:
                score += weight * (1 - abs(movie_stats[qid1][attr] - movie_stats[qid2][attr]))
                matched_attributes.append(attr)

    return score, matched_attributes

# Function to calculate a weighted score based on the proximity of release dates
def calculate_date_proximity(date1, date2, weight):
    try:
        date1 = datetime.strptime(date1, "%Y-%m-%d")
        date2 = datetime.strptime(date2, "%Y-%m-%d")
        delta_days = abs((date1 - date2).days)

        if delta_days <= 365:
            return weight
        elif delta_days <= 1825:  # Up to 5 years difference
            return weight * 0.75
        elif delta_days <= 3650:  # Up to 10 years difference
            return weight * 0.5
        else:
            return weight * 0.25  # More than 10 years difference, minimal contribution
    except (ValueError, TypeError):
        return 0  # If date format is incorrect or parsing fails

# Load movie properties
print("Loading movies from 'movies-tv-shows-properties.json'...")
try:
    with open("../Old Data/movies-properties.json", "r", encoding="utf-8") as f:
        movies_data = json.load(f)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

print(f"Total movies loaded: {len(movies_data)}")

# Load user comments and calculate average sentiment and rating
print("Loading user comments from 'user-comments.csv'...")
movie_stats = load_user_comments("../dataset/user-comments.csv")

# Process each movie and calculate recommendations
recommended_movies = defaultdict(list)

for i, (title1, movie1) in enumerate(tqdm(movies_data.items(), desc="Processing movies", unit=" movie")):
    properties1 = movie1.get("properties", {})
    properties1["qid"] = movie1.get("qid")  # Ensure qid is available for movie_stats lookup

    for j, (title2, movie2) in enumerate(movies_data.items()):
        if title1 == title2:
            continue

        properties2 = movie2.get("properties", {})
        properties2["qid"] = movie2.get("qid")  # Ensure qid is available for movie_stats lookup
        similarity_score, attributes_matched = calculate_similarity(properties1, properties2, movie_stats)

        if similarity_score > 0.3:
            recommended_movies[title1].append({
                "title": title2,
                "similarity_score": similarity_score,
                "average_rating": movie_stats.get(movie2.get("qid"), {}).get("average_rating", "N/A"),
                "attributes_matched": attributes_matched
            })

    # Limit recommendations to the top 5 by similarity score
    recommended_movies[title1] = sorted(recommended_movies[title1], key=lambda x: x["similarity_score"], reverse=True)[:5]

# Save recommendations to movies-similar.json
print("Saving recommendations to 'movies-tv-shows-similar.json'...")
try:
    with open("movies-similar.json", "w", encoding="utf-8") as out_file:
        json.dump(recommended_movies, out_file, indent=2)
    print("Recommendation file 'movies-tv-shows-similar.json' created successfully.")
except Exception as e:
    print(f"Error saving recommendations: {e}")
