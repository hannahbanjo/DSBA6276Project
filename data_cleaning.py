import numpy as np
import pandas as pd

calendarrates = pd.read_csv("")
listings = pd.read_csv("")
reviews = pd.read_csv("")

#Join on listing_id
airroi = calendarrates.merge(listings, left_on='listing_id', right_on='listing_id')
airroi = airroi.merge(reviews, left_on='listing_id', right_on='listing_id')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Check for missing values
print(airroi.isin(['?', '--', 'N/A']).sum())
print(airroi.isna().sum())
print(airroi.info())

#Drop columns for number of null values
airroi = airroi.drop(['instant_book', 'cohost_ids', 'cohost_names'], axis=1)

#Drop rows with missing values
airroi = airroi.dropna(subset=['bedrooms'])
airroi = airroi.dropna(subset=['booking_lead_time_avg'])

#Replace missing values for beds with corresponding value for bedrooms
airroi["beds"] = airroi["beds"].fillna(airroi["bedrooms"])
#Replace missing values for guests with corresponding value for beds
airroi["guests"] = airroi["guests"].fillna(airroi["beds"])

#Extract date
airroi['date_x'] = pd.to_datetime(airroi['date_x'])

airroi['date_day'] = airroi['date_x'].dt.day
airroi['date_month'] = airroi['date_x'].dt.month
airroi['date_year'] = airroi['date_x'].dt.year

airroi = airroi.drop(['date_x'], axis=1)

#Fill missing professional_management as False
airroi['professional_management'] = airroi['professional_management'].fillna(False)

#Convert boolean values into integer types
airroi['professional_management'] = airroi['professional_management'].astype(int)
airroi['superhost'] = airroi['superhost'].astype(int)
airroi['registration'] = airroi['registration'].astype(int)

#Perform dummy coding
airroi = pd.get_dummies(airroi, columns=['listing_type'], prefix='', prefix_sep='', drop_first=True, dtype='int')
airroi = pd.get_dummies(airroi, columns=['cancellation_policy'], prefix='cancellation', prefix_sep='_', drop_first=True, dtype='int')

#Set 0 for missing fee values
airroi['cleaning_fee'] = airroi['cleaning_fee'].fillna(0)
airroi['extra_guest_fee'] = airroi['extra_guest_fee'].fillna(0)
airroi['min_nights'] = airroi['min_nights'].fillna(0)
airroi['min_nights_avg'] = airroi['min_nights_avg'].fillna(0)

#Drop meaningless features
airroi = airroi.drop(['cover_photo_url', 'date_y', 'reviewers', 'currency', 'host_name'], axis=1)

#Download csv
airroi.to_csv('airroi.csv', index=False)
