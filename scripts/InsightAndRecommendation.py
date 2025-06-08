import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re

import emoji

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


class SentimentAnalysis:
    """
    SentimentAnalysis class supports multiple sentiment analysis methods:
    - BERT (transformers pipeline with distilbert-base-uncased-finetuned-sst-2-english)
    - VADER (lexicon and rule-based)
    - TextBlob (lexicon-based)

    It provides text preprocessing (emoji removal, non-English filtering, tokenization,
    stop word removal, lemmatization), sentiment prediction, adding sentiment column to DataFrame,
    and aggregation of sentiment counts by bank and rating.

    Attributes:
        method (str): One of 'bert', 'vader', or 'textblob' specifying the sentiment analysis method.
    """

    def __init__(self, method="bert"):
        """
        Initialize the SentimentAnalysis class.

        Args:
            method (str): Sentiment analysis method to use. Options: 'bert', 'vader', 'textblob'.
                          Default is 'bert'.

        Raises:
            ValueError: If the method is not one of the supported options.
        """
        self.method = method.lower()

        if self.method == "bert":
            self.analyzer = pipeline(
                "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        elif self.method == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.method == "textblob":
            self.analyzer = None  # TextBlob does not require initialization
        else:
            raise ValueError(f"Unsupported method '{method}'. Choose from 'bert', 'vader', or 'textblob'.")

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    def extract_emoji_and_non_english(self, df: pd.DataFrame, text_col='review') -> dict:
            """
            Extract reviews containing emojis and non-English reviews by bank.
            
            Returns:
                {
                    'emoji': pd.DataFrame with columns ['bank', 'review'] of reviews containing emojis,
                    'non_english': pd.DataFrame with columns ['bank', 'review'] of non-English reviews
                }
            """
            df = df.copy()

            def has_emoji(text):
                return bool(emoji.emoji_count(text)) if isinstance(text, str) else False

            def is_non_english(text):
                if not isinstance(text, str) or not text.strip():
                    return False
                try:
                    return detect(text) != 'en'
                except LangDetectException:
                    return False

            df['has_emoji'] = df[text_col].apply(has_emoji)
            df['non_english'] = df[text_col].apply(is_non_english)

            emoji_reviews = df[df['has_emoji']][['bank', text_col]].rename(columns={text_col: 'review'})
            non_english_reviews = df[df['non_english']][['bank', text_col]].rename(columns={text_col: 'review'})

            return {
                'emoji': emoji_reviews.reset_index(drop=True),
                'non_english': non_english_reviews.reset_index(drop=True)
            }
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by:
        - Removing emojis
        - Filtering out non-English text
        - Lowercasing
        - Removing punctuation and digits
        - Tokenizing
        - Removing stopwords and lemmatizing (for 'vader' and 'textblob' methods)

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Cleaned and preprocessed text. Returns empty string if input is invalid or non-English.
        """
        if not isinstance(text, str):
            return ""

        # Remove emojis
        text = emoji.replace_emoji(text, replace='')

        # Filter out non-English text
        try:
            if detect(text) != "en":
                return ""
        except Exception:
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation and digits
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords and lemmatize for certain methods
        if self.method in ['vader', 'textblob']:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]

        # Reconstruct cleaned text
        return " ".join(tokens)

    def get_sentiment(self, text: str) -> str:
        """
        Predict sentiment label for input text.

        Args:
            text (str): Input text.

        Returns:
            str: Sentiment label: 'positive', 'neutral', or 'negative'.
        """
        if not isinstance(text, str) or not text.strip():
            return "neutral"

        if self.method == "bert":
            # Limit input length to 512 tokens for BERT
            result = self.analyzer(text[:512])[0]
            label = result['label'].lower()
            return "positive" if label == "positive" else "negative"

        elif self.method == "vader":
            score = self.analyzer.polarity_scores(text)["compound"]
            if score >= 0.05:
                return "positive"
            elif score <= -0.05:
                return "negative"
            else:
                return "neutral"

        elif self.method == "textblob":
            score = TextBlob(text).sentiment.polarity
            if score >= 0.1:
                return "positive"
            elif score <= -0.1:
                return "negative"
            else:
                return "neutral"

        return "neutral"

    def add_sentiment_column(self, df: pd.DataFrame, text_col='review') -> pd.DataFrame:
        """
        Add columns for cleaned text and sentiment labels to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing text data.
            text_col (str): Name of the column containing text to analyze.

        Returns:
            pd.DataFrame: New DataFrame with 'cleaned_review' and 'sentiment' columns added.
        
        Raises:
            ValueError: If `text_col` does not exist in DataFrame.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        df = df.copy()
        df['cleaned_review'] = df[text_col].apply(self.preprocess_text)
        df['sentiment'] = df['cleaned_review'].apply(self.get_sentiment)
        return df

    def aggregate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment counts by 'bank' and 'rating'.

        Args:
            df (pd.DataFrame): DataFrame containing at least 'bank', 'rating', and 'sentiment' columns.

        Returns:
            pd.DataFrame: Aggregated sentiment counts with columns for each sentiment label.

        Raises:
            ValueError: If required columns are missing in DataFrame.
        """
        required_cols = {'app_id', 'score', 'sentiment'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        return (
            df.groupby(['app_id', 'score', 'sentiment'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
    def extract_keywords_tfidf_by_app(
        self,
        df: pd.DataFrame,
        text_col='cleaned_review',
        top_n=20,
        ngram_range=(1, 2)
    ) -> pd.DataFrame:
        """
        Extract significant keywords or n-grams using TF-IDF per app_id.

        Args:
            df (pd.DataFrame): DataFrame containing cleaned text and app_id.
            text_col (str): Column name of the cleaned review text.
            top_n (int): Number of top keywords or n-grams to return per app_id.
            ngram_range (tuple): Range of n-grams (e.g., (1,2) for unigrams and bigrams).

        Returns:
            pd.DataFrame: DataFrame with columns ['app_id', 'term', 'score'] showing top keywords per app.
        """
        if text_col not in df.columns or 'app_id' not in df.columns:
            raise ValueError("DataFrame must contain 'app_id' and the specified text column.")

        result_frames = []

        for app_id, group in df.groupby('app_id'):
            texts = group[text_col].dropna().tolist()

            if not texts:
                continue  # Skip empty app_ids

            vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=10000)
            tfidf_matrix = vectorizer.fit_transform(texts)

            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            terms = vectorizer.get_feature_names_out()

            scores_df = pd.DataFrame({
                'app_id': app_id,
                'term': terms,
                'score': tfidf_scores
            })

            top_terms = scores_df.sort_values(by='score', ascending=False).head(top_n)
            result_frames.append(top_terms)

        # Combine all app-specific results into one DataFrame
        final_df = pd.concat(result_frames, ignore_index=True)
        return final_df
    def group_keywords_by_theme(self,keywords_df, themes):
        grouped_by_app = {}

        for app_id in keywords_df['app_id'].unique():
            subset = keywords_df[keywords_df['app_id'] == app_id]
            grouped = defaultdict(list)

            for _, row in subset.iterrows():
                keyword = row['term'].lower()
                score = row['score']
                assigned = False

                for theme, terms in themes.items():
                    if any(re.search(rf"\b{re.escape(term)}\b", keyword) for term in terms):
                        grouped[theme].append((keyword, score))
                        assigned = True
                        break

                if not assigned:
                    grouped["Other"].append((keyword, score))

            grouped_by_app[app_id] = grouped

        return grouped_by_app