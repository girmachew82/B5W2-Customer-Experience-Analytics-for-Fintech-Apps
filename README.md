# B5W2-Customer-Experience-Analytics-for-Fintech-Apps
# Task 1
# Bank App Review Scraper & Sentiment Analysis

This project collects, cleans, and analyzes user reviews from Google Play Store for selected Ethiopian banking apps. It enables data-driven insight into customer feedback using NLP and sentiment analysis techniques.

## ✅ Targeted Apps
- Dashen Bank (com.dashen.dashensuperapp)
- Commercial Bank of Ethiopia (com.combanketh.mobilebanking)
- Bank of Abyssinia (com.boa.boaMobileBanking)

---

## 🔍 Methodology

### 1. Data Collection
- Used [`google-play-scraper`](https://github.com/JoMingyu/google-play-scraper) to fetch app reviews.
- Each app's reviews were collected using:
  ```python
  reviews_all(app_id, sleep_milliseconds, lang, country, sort, filter_score_with)
  ```
- Export all banks review data to a single csv:
```python
  obj.export_all_reviews_to_single_csv(
    reviews_list=[
        ("com.combanketh.mobilebanking", cbe_reviews),
        ("com.boa.boaMobileBanking", boa_reviews),
        ("com.dashen.dashensuperapp", dashen_bank_reviews),
    ],
    csv_filename="all_bank_reviews.csv"
)
```
- Read as pandas data
```
df = pd.read_csv("data/all_bank_reviews.csv")
```
- Clean data like missing value and duplicate 
```
print(df.isnull().sum())
df = df.dropna(subset=['content', 'score'])
```
- Save cleaned data
```
df.to_csv("data/cleaned_bank_reviews.csv", index=False)
```
- Date format 
```
df['at'] = pd.to_datetime(df['at'], errors='coerce').dt.strftime('%Y-%m-%d')
```
- Save date formated data
```
df.to_csv("data/cleaned_bank_reviews.csv", index=False)
```
- Column name change
```
df_final = df.rename(columns={
    'content': 'review',
    'score': 'rating',
    'at': 'date',
    'app_id': 'bank'
})
```
- Add Data source 
```
df_final['source'] = 'Google Play Store'
```
- Keep only relevent columns
```
df_final = df_final[['review', 'rating', 'date', 'bank', 'source']]
```
- Save new CSV data
```
df_final.to_csv("data/final_bank_reviews.csv", index=False)
```
# Task 2 Sentiment and Thematic Analysis
## 🌐 Methods
- Preprocessing 
  - Removing emojis
  - Filtering out non-English text
  - Lowercasing
  - Removing punctuation and digits
  - Tokenizing
  - Removing stopwords and lemmatizing (for 'vader' and 'textblob' methods)

- Get sentement
```
- Neutral
- Positive
- Negative
```
- Aggregate Sentiment
```
Group by bank id, score and sentiment
```
- Extract keywords by tfidf
- Group keywords by theme