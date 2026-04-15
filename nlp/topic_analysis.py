import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv('nlp/outputs/listings_with_features_week2.csv')

# define segment mapping
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

# grouping into strategies
experience_topics = [4, 7, 2, 0, 8]
functional_topics = [3, 1, 6, 9, 5]

def get_strategy(topic_id):
    if topic_id in experience_topics:
        return "Experience Strategy"
    return "Functional Strategy"

# calculate metrics per topic
topic_stats = df.groupby('dominant_topic')[['ttm_revenue', 'ttm_avg_rate', 'ttm_occupancy']].mean().reset_index()
topic_stats['topic_name'] = topic_stats['dominant_topic'].map(segment_map)
topic_stats['strategy'] = topic_stats['dominant_topic'].apply(get_strategy)

soft_palette = {"Experience Strategy": "#8da0cb", "Functional Strategy": "#fc8d62"}

sns.set_style("whitegrid")

# revenue chart
rev_data = topic_stats.sort_values(by='ttm_revenue', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='ttm_revenue', y='topic_name', data=rev_data, hue='strategy', palette=soft_palette, dodge=False)
plt.title('Average Annual Revenue (TTM) by Segment', fontsize=16, fontweight='bold')
for i, v in enumerate(rev_data['ttm_revenue']):
    plt.text(v, i, f' ${v:,.0f}', va='center', fontweight='bold')
plt.legend(title='Positioning Strategy', loc='lower right')
plt.tight_layout()
plt.savefig('nlp/visualizations/revenue_by_topic.png')

# rate chart
rate_data = topic_stats.sort_values(by='ttm_avg_rate', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='ttm_avg_rate', y='topic_name', data=rate_data, hue='strategy', palette=soft_palette, dodge=False)
plt.title('Average Daily Rate (ADR) by Segment', fontsize=16, fontweight='bold')
for i, v in enumerate(rate_data['ttm_avg_rate']):
    plt.text(v, i, f' ${v:,.0f}', va='center', fontweight='bold')
plt.legend(title='Positioning Strategy', loc='lower right')
plt.tight_layout()
plt.savefig('nlp/visualizations/rate_by_topic.png')

# occupancy chart
occ_data = topic_stats.sort_values(by='ttm_occupancy', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='ttm_occupancy', y='topic_name', data=occ_data, hue='strategy', palette=soft_palette, dodge=False)
plt.title('Average Annual Occupancy by Segment', fontsize=16, fontweight='bold')
for i, v in enumerate(occ_data['ttm_occupancy']):
    plt.text(v, i, f' {v*100:.1f}%', va='center', fontweight='bold')
plt.legend(title='Positioning Strategy', loc='lower right')
plt.tight_layout()
plt.savefig('nlp/visualizations/occupancy_by_topic.png')