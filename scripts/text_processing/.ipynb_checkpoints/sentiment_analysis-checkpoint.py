import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from scripts.text_processing.preprocess_tweets_lite import TextCleaner
import pandas as pd

class SentimentAnalyzer(TextCleaner):
    def __init__(self, model="cardiffnlp/twitter-roberta-base-sentiment-latest", emotion=False):
        super().__init__(stop_words_remove=False)
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.MODEL = model
        self.load_model()
        self.emotion = emotion

    def load_model(self):
        MODEL = self.MODEL
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def get_sentiment(self, df):
        _df = df.copy()
        _df["cleaned_text"] = _df['text'].apply(self.clean_text)
        cleaned_text = _df["cleaned_text"].values.tolist()
        # Truncate or pad the input text to a consistent length
        max_length = 512  # You can adjust this value according to your needs
        results = self.classifier(cleaned_text, padding=True, truncation=True, max_length=max_length)
        
        if self.emotion == True:
            _df["emotion"] = [result['label'] for result in results]
            _df["emotion_score"] = [result['score'] for result in results]
        else:
            _df["sentiment"] = [result['label'] for result in results]
            _df["sentiment_score"] = [result['score'] for result in results]
        return _df
