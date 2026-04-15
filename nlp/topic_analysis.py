import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv('nlp/outputs/listings_with_features_week2.csv')

# define segment mapping
segment_map = {
    4: "Luxury Lifestyle", 
    7: "Corporate Executive",
    3: "Family Hubs",
    2: "Premium Multifamily",
    0: "Upscale Urban",
    8: "Modern Condos",
    1: "Core Residential",
    6: "Mid-Market Utility",
    9: "Traditional Neighborhood",
    5: "Budget Studios"
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

# Set style to white to remove the grey background
sns.set_style("white")

# revenue chart
rev_data = topic_stats.sort_values(by='ttm_revenue', ascending=False)
plt.figure(figsize=(8, 5)) 
ax = sns.barplot(x='ttm_revenue', y='topic_name', data=rev_data, hue='strategy', palette=soft_palette, dodge=False)
plt.title('Average Annual Revenue (TTM) by Segment', fontsize=12, fontweight='bold')
plt.xlabel('Revenue ($)', fontsize=10)
plt.ylabel('')
plt.yticks(fontsize=9)

# Remove all grid lines and spine lines
ax.grid(False)
sns.despine(left=False, bottom=False)

for i, v in enumerate(rev_data['ttm_revenue']):
    plt.text(v + 1000, i, f' ${v:,.0f}', va='center', ha='left', fontsize=9, fontweight='bold')

plt.legend(title='Strategy', fontsize=8, title_fontsize=9, loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False)
plt.subplots_adjust(left=0.25, right=0.7)
plt.savefig('nlp/visualizations/revenue_by_topic.png', transparent=True, bbox_inches='tight')

# rate chart
rate_data = topic_stats.sort_values(by='ttm_avg_rate', ascending=False)
plt.figure(figsize=(8, 5))
ax = sns.barplot(x='ttm_avg_rate', y='topic_name', data=rate_data, hue='strategy', palette=soft_palette, dodge=False)
plt.title('Average Daily Rate (ADR) by Segment', fontsize=12, fontweight='bold')
plt.xlabel('Rate ($)', fontsize=10)
plt.ylabel('')
plt.yticks(fontsize=9)

ax.grid(False)
sns.despine(left=False, bottom=False)

for i, v in enumerate(rate_data['ttm_avg_rate']):
    plt.text(v + 5, i, f' ${v:,.0f}', va='center', ha='left', fontsize=9, fontweight='bold')

plt.legend(title='Strategy', fontsize=8, title_fontsize=9, loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False)
plt.subplots_adjust(left=0.25, right=0.7)
plt.savefig('nlp/visualizations/rate_by_topic.png', transparent=True, bbox_inches='tight')

# occupancy chart
occ_data = topic_stats.sort_values(by='ttm_occupancy', ascending=False)
plt.figure(figsize=(8, 5))
ax = sns.barplot(x='ttm_occupancy', y='topic_name', data=occ_data, hue='strategy', palette=soft_palette, dodge=False)
plt.title('Average Annual Occupancy by Segment', fontsize=12, fontweight='bold')
plt.xlabel('Occupancy (%)', fontsize=10)
plt.ylabel('')
plt.yticks(fontsize=9)

ax.grid(False)
sns.despine(left=False, bottom=False)

for i, v in enumerate(occ_data['ttm_occupancy']):
    plt.text(v + 0.01, i, f' {v*100:.1f}%', va='center', ha='left', fontsize=9, fontweight='bold')

plt.legend(title='Strategy', fontsize=8, title_fontsize=9, loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False)
plt.subplots_adjust(left=0.25, right=0.7)
plt.savefig('nlp/visualizations/occupancy_by_topic.png', transparent=True, bbox_inches='tight')