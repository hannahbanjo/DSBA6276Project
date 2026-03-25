import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('nlp/outputs/listings_with_features_week2.csv')

print(df.info())
print(df.head())

# calculate correlations of the new features with target metrics
target_metrics = ['ttm_occupancy', 'ttm_revenue', 'ttm_avg_rate']
feature_cols = [col for col in df.columns if col.startswith('sent_') or col.startswith('title_has_') or col == 'dominant_topic']

# if dominant_topic is categorical, we might need a different approach, but for now let's see correlations for continuous ones
continuous_features = [col for col in df.columns if col.startswith('sent_') or col.startswith('title_has_')]
correlation_matrix = df[continuous_features + target_metrics].corr().loc[continuous_features, target_metrics]

print("\nCorrelations with Target Metrics:")
print(correlation_matrix)

# check topic distribution
if 'dominant_topic' in df.columns:
    print("\nTopic Distribution:")
    print(df['dominant_topic'].value_counts())
    
    # calculate metrics per topic
    topic_summary = df.groupby('dominant_topic')[target_metrics].mean()
    print("\nMetrics by Topic:")
    print(topic_summary)


# visualize topic performance
segment_map = {
    4: "Luxury Lifestyle Curations",
    7: "Corporate Executive Residences",
    3: "Extended-Stay Family Hubs",
    2: "Premium Multifamily Assets",
    0: "Upscale Urban Professional",
    8: "Modern Amenity-Forward Condos",
    1: "Core Residential Essentials",
    6: "Mid-Market Utility Units",
    9: "Traditional Neighborhood Stock",
    5: "Minimalist Budget Studios"
}

topic_stats = df.groupby('dominant_topic')[['ttm_revenue', 'ttm_avg_rate', 'ttm_occupancy']].mean().reset_index()
topic_stats['topic_name'] = topic_stats['dominant_topic'].map(segment_map)

sns.set_style("whitegrid")

# revenue chart
rev_data = topic_stats.sort_values(by='ttm_revenue', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='ttm_revenue', y='topic_name', data=rev_data, hue='topic_name', palette='Set2', legend=False)
plt.title('Average Annual Revenue (TTM) by Segment', fontsize=16, fontweight='bold')
for i, v in enumerate(rev_data['ttm_revenue']):
    plt.text(v, i, f' ${v:,.0f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('nlp/visualizations/revenue_by_topic.png')

# rate chart
rate_data = topic_stats.sort_values(by='ttm_avg_rate', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='ttm_avg_rate', y='topic_name', data=rate_data, hue='topic_name', palette='Set2', legend=False)
plt.title('Average Daily Rate (ADR) by Segment', fontsize=16, fontweight='bold')
for i, v in enumerate(rate_data['ttm_avg_rate']):
    plt.text(v, i, f' ${v:,.0f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('nlp/visualizations/rate_by_topic.png')

# occupancy chart
occ_data = topic_stats.sort_values(by='ttm_occupancy', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='ttm_occupancy', y='topic_name', data=occ_data, hue='topic_name', palette='Set2', legend=False)
plt.title('Average Annual Occupancy by Segment', fontsize=16, fontweight='bold')
for i, v in enumerate(occ_data['ttm_occupancy']):
    # Formats 0.596 as 59.6%
    plt.text(v, i, f' {v*100:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('nlp/visualizations/occupancy_by_topic.png')