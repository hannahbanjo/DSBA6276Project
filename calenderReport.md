# Week 1: Temporal Feature Engineering & Demand Segmentation

## Data Preparation  
The dataset was constructed by combining both historical (`past_calendar_rates.csv`) and forecasted (`future_calendar_rates.csv`) calendar data into a unified dataset. The `date` column was converted into a datetime format, and additional temporal features such as `month` and `year` were extracted to support seasonal analysis.

---

## Demand Period Definition  
To capture seasonal demand patterns, a categorical variable `demand_period` was created based on the month of each observation. The classification is as follows:

- **Peak:** June–August and December  
- **Shoulder:** April–May and September–October  
- **Off-Peak:** January–March and November  

This segmentation is intended to reflect typical travel and booking seasonality.

---

## Exploratory Analysis  

### Occupancy Trends  
Average occupancy rates were found to be relatively consistent across all demand periods:

- Off-peak: 0.251  
- Peak: 0.246  
- Shoulder: 0.253  

This suggests that occupancy does not vary significantly by season in the current dataset.

---

### Revenue Trends  
Revenue shows clearer variation across demand periods:

- Off-peak: $1,206.60  
- Peak: $1,314.16  
- Shoulder: $1,400.59  

Interestingly, **shoulder periods generate the highest average revenue**, exceeding even peak periods.

---

### Pricing Trends  
Average listing rates also vary across demand periods:

- Off-peak: $170.20  
- Peak: $182.01  
- Shoulder: $179.44  

Peak periods have the highest pricing on average, followed closely by shoulder periods.

---

## Key Insights  

1. **Revenue is highest during shoulder periods**, suggesting a balance between strong demand and pricing strategy.  
2. **Peak periods have the highest prices**, but not the highest revenue.  
3. **Occupancy remains stable across all periods**, indicating pricing plays a larger role in revenue differences.  

---

## Conclusion  

The demand period segmentation provides a useful framework for understanding seasonal trends in pricing and revenue. While occupancy remains stable, variations in pricing and revenue highlight the importance of strategic rate adjustments across different demand periods. These engineered temporal features will support further modeling in Week 2, particularly in analyzing the impact of superhost status and seasonal demand on performance metrics.