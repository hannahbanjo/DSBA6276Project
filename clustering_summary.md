# DSBA6276Project Initial Clustering

- Source file: `/Users/laasyavenugopal/Desktop/Masters/Spring 2026/Strategic Business Analytics/Group Project DSBA6276/airroi.csv`
- Raw rows in source file: 16552
- Listing-level rows used: 220
- Selected clustering features: guests, bedrooms, baths, cleaning_fee, num_reviews_x, rating_overall, superhost, professional_management, ttm_avg_rate, ttm_occupancy
- Log-transformed before scaling: cleaning_fee, num_reviews_x, ttm_avg_rate
- Optimal k from silhouette scan (2 to 8): 2

## Notes

- One row per listing is used so repeated monthly snapshots do not dominate the clusters.
- Cluster labels are numeric and re-ordered deterministically so the initial labels stay locked.
- Outputs include the attribute inventory, feature table, k-evaluation table, final assignments, and cluster profiles.