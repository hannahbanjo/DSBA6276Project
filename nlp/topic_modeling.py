from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

corrs = pd.read_csv('nlp/outputs/term_performance_correlations.csv')

# 1. Find words highly correlated with high occupancy (Potential "Cleanliness/Value" indicators)
top_performers = corrs.sort_values(by='ttm_occupancy', ascending=False).head(20)['term'].tolist()

# 2. Find words highly correlated with high revenue (Potential "Luxury/Location" indicators)
luxury_indicators = corrs.sort_values(by='ttm_revenue', ascending=False).head(20)['term'].tolist()

print(f"Suggested for Cleanliness/Service: {top_performers}")
print(f"Suggested for Location/Amenities: {luxury_indicators}")

analyzer = SentimentIntensityAnalyzer()
df = pd.read_csv('nlp/outputs/listings_with_corpus.csv')

#dimension keywords based on correlation results
dimensions = {
    'cleanliness_service': [
        'dishes', 'silverware', 'linens', 'essentials', 
        'clean', 'spotless', 'fresh', 'towels'
    ],
    'amenities_value': [
        'kitchen', 'microwave', 'refrigerator', 'dishwasher', 
        'fire_pit', 'pool', 'cooking_basics', 'board_games'
    ],
    'location_vibe': [
        'downtown', 'bungalow', 'uptown', 'dilworth', 
        'southend', 'walkable', 'close', 'staycation'
    ],
    'high_end_features': [
        'smart', 'loft', 'bathtub', 'bidet', 
        'safe', 'outdoor_dining', 'hammock'
    ]
}

def get_dimension_sentiment(text, keywords):
    # extract sentences containing keywords to isolate sentiment
    relevant_sentences = [sent for sent in str(text).split('.') if any(kw in sent.lower() for kw in keywords)]

    if  relevant_sentences:
        return 0.0  # neutral
    return analyzer.polarity_scores(' '.join(relevant_sentences))['compound']

for dim, keywords in dimensions.items():
    df[f'sent_{dim}'] = df['text_corpus'].apply(lambda x: get_dimension_sentiment(x, keywords))

# global sentiment
df['sentiment_overall'] = df['text_corpus'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# topic modeling
vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df['text_corpus'].astype(str))

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(doc_term_matrix)

topic_values = lda.transform(doc_term_matrix)
df['dominant_topic'] = topic_values.argmax(axis=1)

def print_topics(model, vecotrizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic {idx}: {[vecotrizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n]]}")

print_topics(lda, vectorizer)

# listing title keyword extraction
target_keywords = ['luxury', 'cozy', 'walkable', 'pet-friendly', 'modern', 'spacious']

for kw in target_keywords:
    df[f'title_has_{kw.replace("-", "_")}'] = df['listing_name'].str.contains(kw, case=False, na=False).astype(int)

df.to_csv('nlp/outputs/listings_with_features_week2.csv', index=False)