import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("/Users/hannahw/DSBA6276/DSBA6276Project/airroi.csv")

# build text corpus
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['listing_name_clean'] = df['listing_name'].apply(clean_text)

def parse_amenities(val):
    if pd.isna(val) or str(val).strip() == "":
        return ''
    
    items = [a.strip() for a in str(val).split(',') if a.strip()]
    normalized = []

    for a in items:
        a = a.lower()
        a = re.sub(r'[^a-z0-9\s]', '', a)
        a = re.sub(r'\s+', '_', a).strip()
        normalized.append(a)

    return ' '.join(normalized)

df['amenities_text'] = df['amenities'].apply(parse_amenities)

df['text_corpus'] = df['listing_name_clean'] + ' ' + df['amenities_text']

print("\nSample corpus documents:")
print(df[['listing_id', 'room_type', 'text_corpus']].head(3).to_string())

# split by room type
entire_house = df["room_type"] == "entire_home"
private_room = df["room_type"] == "private_room"

entire_house_corpus = df[entire_house]['text_corpus'].tolist()
private_room_corpus = df[private_room]['text_corpus'].tolist()

print(f"\nEntire place listings: {len(entire_house_corpus)}")
print(f"Private room listings: {len(private_room_corpus)}")

# tfidf vectorization
def fit_tfidf(corpus, label, top_n=30):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.90,
        min_df=2,           
        max_features=3000
    )

    matrix = vectorizer.fit_transform(corpus)
    mean_scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    top_idx = mean_scores.argsort()[::-1][:top_n]
    results = pd.DataFrame({
        'term' : terms[top_idx],
        'tdidf_score' : mean_scores[top_idx].round(4),
        'segment' : label
    })

    print(f"\nTop {top_n} terms for {label}:")
    print(results[['term', 'tdidf_score']].to_string(index=False))

    return results, vectorizer

entire_house_terms, entire_house_vectorizer = fit_tfidf(entire_house_corpus, 'entire_place')
private_room_terms, private_room_vectorizer = fit_tfidf(private_room_corpus, 'private_room')

# find distinctive terms per segment
merged = entire_house_terms.merge(
    private_room_terms[['term', 'tdidf_score']],
    on='term',
    how='outer',
    suffixes=('_entire', '_private')
).fillna(0)

merged['distinctive_entire'] = merged['tdidf_score_entire'] - merged['tdidf_score_private']
merged['distinctive_private'] = merged['tdidf_score_private'] - merged['tdidf_score_entire']

print("\n Most distinctive for Entire place")
print(merged.nlargest(15, 'distinctive_entire')[
    ['term', 'tdidf_score_entire', 'tdidf_score_private', 'distinctive_entire']
].to_string(index=False))

print("\nMost distinctive for Private room")
print(merged.nlargest(15, 'distinctive_private')[
    ['term', 'tdidf_score_private', 'tdidf_score_entire', 'distinctive_private']
].to_string(index=False))

# correlate with performance outcomes
vectorizer_full = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    max_df=0.90,
    min_df=3,
    max_features=3000
)

tdidf_matrix = vectorizer_full.fit_transform(df['text_corpus'])
tf_idf_df = pd.DataFrame(
    tdidf_matrix.toarray(),
    columns=vectorizer_full.get_feature_names_out(),
    index=df.index)


outcomes = df[['ttm_occupancy', 'ttm_avg_rate', 'ttm_revenue']]

corr_results = {}

for outcome in outcomes:
    corr_results[outcome] = tf_idf_df.corrwith(outcomes[outcome])

corr_df = pd.DataFrame(corr_results)
corr_df.index.name = 'term'
corr_df = corr_df.reset_index()

# top 15 positive for occupancy
print("\nTerms most correlated with HIGH occupancy")
print(corr_df.nlargest(15, 'ttm_occupancy')[
    ['term', 'ttm_occupancy', 'ttm_avg_rate']
].to_string(index=False))

# top 15 negative for occupancy (vacancy risk signal)
print("\nTerms most correlated with LOW occupancy")
print(corr_df.nsmallest(15, 'ttm_occupancy')[
    ['term', 'ttm_occupancy', 'ttm_avg_rate']
].to_string(index=False))

# terms that command higher rates
print("\nTerms most correlated with HIGH avg rate")
print(corr_df.nlargest(15, 'ttm_avg_rate')[
    ['term', 'ttm_avg_rate', 'ttm_occupancy']
].to_string(index=False))


# run correlations separately by room type
for segment in ['entire_home', 'private_room']:
    seg = (df['room_type'] == segment).values
    segment_tfidf = tf_idf_df[seg]

    segment_outcomes = df[seg]

    corr_occ = segment_tfidf.corrwith(segment_outcomes['ttm_occupancy'])
    corr_rate = segment_tfidf.corrwith(segment_outcomes['ttm_avg_rate'])

    corr_df = corr_df.dropna()

    seg_corr = pd.DataFrame({
        'term' : corr_occ.index,
        'corr_occupancy' : corr_occ.values,
        'corr_avg_rate' : corr_rate.values,
        'segment' : segment
    }).sort_values('corr_occupancy', ascending=False)


    print(f"\n── Top occupancy signals: {segment} ──")
    print(seg_corr.head(10)[['term', 'corr_occupancy', 'corr_avg_rate']].to_string(index=False))

# save results
entire_house_terms.to_csv('nlp/outputs/tfidf_entire.csv', index=False)
private_room_terms.to_csv('nlp/outputs/tfidf_private.csv', index=False)
merged.to_csv('nlp/outputs/tfidf_comparison.csv', index=False)

# full correlation table - useful for Week 2 when we build a predictive model
corr_df.to_csv('nlp/outputs/term_performance_correlations.csv', index=False)

# save full dataframe with corpus for future reference
df.to_csv('nlp/outputs/listings_with_corpus.csv', index=False)

print("\nAll outputs saved. Week 1 complete.")

