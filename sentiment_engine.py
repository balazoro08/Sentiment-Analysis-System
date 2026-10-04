import re
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from dataset_generator import get_combined_dataset

# Domain-specific sentiment lexicons for enhanced precision
POSITIVE_LEXICON = {
    'outstanding', 'fantastic', 'excellent', 'amazing', 'superb', 'awesome', 'brilliant',
    'perfect', 'love', 'loved', 'great', 'good', 'impressive', 'incredible', 'best',
    'top', 'seamless', 'flawless', 'satisfied', 'worth', 'helpful', 'useful', 'intuitive',
    'durable', 'fast', 'reliable', 'smooth', 'recommend', 'recommended', 'exceeded',
    'craftsmanship', 'premium', 'delightful', 'work', 'works', 'lifesaver', 'clean', 'polished'
}

NEGATIVE_LEXICON = {
    'terrible', 'horrible', 'awful', 'disappointing', 'disappointed', 'poor', 'defective',
    'broken', 'junk', 'waste', 'ruined', 'useless', 'glitch', 'glitches', 'crashes', 'crash',
    'freeze', 'freezes', 'overpriced', 'cheap', 'flimsy', 'slow', 'overheating', 'nightmare',
    'frustrating', 'frustrated', 'unusable', 'scratched', 'damaged', 'missing', 'unhelpful',
    'drains', 'popups', 'annoying', 'intrusive', 'regret', 'loud', 'fail', 'fails', 'failed'
}

NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'nor', 'cannot', "can't", 'don\'t', 'doesn\'t', 'didn\'t', 'without', 'hardly', 'barely'}


class SentimentEngine:
    """Hybrid NLP & Machine Learning Engine for Sentiment Classification."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(C=2.0, max_iter=1000, solver='lbfgs')
        self.classes_ = ['negative', 'neutral', 'positive']
        self.is_trained = False
        self.model_metrics = {}
        self.train_model()

    def preprocess_text(self, text: str) -> str:
        """Clean and transform text with negation awareness."""
        if not text:
            return ""
        
        text_lower = text.lower()
        # Normalize contractions
        text_lower = re.sub(r"n't", " not", text_lower)
        text_lower = re.sub(r"'re", " are", text_lower)
        text_lower = re.sub(r"'s", " is", text_lower)
        text_lower = re.sub(r"'d", " would", text_lower)
        text_lower = re.sub(r"'ll", " will", text_lower)
        text_lower = re.sub(r"'t", " not", text_lower)
        text_lower = re.sub(r"'ve", " have", text_lower)
        text_lower = re.sub(r"'m", " am", text_lower)
        
        # Tokenize preserving punctuation for negation tagging
        tokens = re.findall(r"\w+|[^\w\s]", text_lower)
        
        processed_tokens = []
        negate = False
        for token in tokens:
            if token in ['.', '!', '?', ';', ',']:
                negate = False
                processed_tokens.append(token)
                continue
                
            if token in NEGATION_WORDS:
                negate = True
                processed_tokens.append(token)
                continue
                
            if negate and re.match(r'^[a-z]+$', token):
                processed_tokens.append(f"NOT_{token}")
            else:
                processed_tokens.append(token)
                
        return " ".join(processed_tokens)

    def compute_rule_polarity(self, text: str) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Calculates rule-based polarity (-1 to 1) and subjectivity (0 to 1)."""
        words = re.findall(r'\b[a-z_]+\b', text.lower())
        pos_count = 0
        neg_count = 0
        subjective_count = 0
        token_highlights = []

        raw_words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text)
        
        for word in raw_words:
            w_lower = word.lower()
            tag = "neutral"
            score = 0.0

            is_negated = w_lower.startswith("not_")
            clean_w = w_lower.replace("not_", "")

            if clean_w in POSITIVE_LEXICON:
                if is_negated:
                    neg_count += 1.2
                    tag = "negative"
                    score = -0.8
                else:
                    pos_count += 1.0
                    tag = "positive"
                    score = 0.8
                subjective_count += 1

            elif clean_w in NEGATIVE_LEXICON:
                if is_negated:
                    pos_count += 1.0
                    tag = "positive"
                    score = 0.7
                else:
                    neg_count += 1.2
                    tag = "negative"
                    score = -0.8
                subjective_count += 1
            
            token_highlights.append({
                "word": word,
                "sentiment": tag,
                "score": score
            })

        total_eval = pos_count + neg_count
        if total_eval == 0:
            polarity = 0.0
        else:
            polarity = (pos_count - neg_count) / (total_eval + 0.5)

        total_words = max(len(raw_words), 1)
        subjectivity = min(1.0, (subjective_count + 0.5 * (pos_count + neg_count)) / total_words)

        return float(polarity), float(subjectivity), token_highlights

    def train_model(self, custom_df: pd.DataFrame = None):
        """Train the TF-IDF + Classifier pipeline."""
        if custom_df is None or custom_df.empty:
            df = get_combined_dataset()
        else:
            df = custom_df.copy()

        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['sentiment'].values

        self.classifier.fit(X, y)
        self.is_trained = True

        # Calculate performance metrics
        y_pred = self.classifier.predict(X)
        acc = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y, y_pred, labels=self.classes_).tolist()
        
        # Get top feature weights per class
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_positive = []
        top_negative = []

        if hasattr(self.classifier, 'coef_'):
            # Multi-class or binary coef
            coefs = self.classifier.coef_
            pos_idx = list(self.classifier.classes_).index('positive') if 'positive' in self.classifier.classes_ else 0
            neg_idx = list(self.classifier.classes_).index('negative') if 'negative' in self.classifier.classes_ else 0

            top_pos_indices = np.argsort(coefs[pos_idx])[-15:][::-1]
            top_neg_indices = np.argsort(coefs[neg_idx])[-15:][::-1]

            top_positive = [{"word": feature_names[i], "weight": round(float(coefs[pos_idx][i]), 3)} for i in top_pos_indices]
            top_negative = [{"word": feature_names[i], "weight": round(float(coefs[neg_idx][i]), 3)} for i in top_neg_indices]

        self.model_metrics = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "confusion_matrix": cm,
            "classes": self.classes_,
            "sample_count": len(df),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative
        }
        return self.model_metrics

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform sentiment prediction and token analysis for a single text."""
        if not text or not text.strip():
            return {
                "text": "",
                "sentiment": "neutral",
                "confidence": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "probabilities": {"positive": 33.3, "neutral": 33.4, "negative": 33.3},
                "tokens": []
            }

        cleaned = self.preprocess_text(text)
        tfidf_vec = self.vectorizer.transform([cleaned])
        
        # Get ML probabilities
        ml_probs = self.classifier.predict_proba(tfidf_vec)[0]
        prob_dict = {cls: float(prob) for cls, prob in zip(self.classifier.classes_, ml_probs)}

        # Ensure all 3 classes exist
        for cls in self.classes_:
            if cls not in prob_dict:
                prob_dict[cls] = 0.0

        # Rule-based overlay
        polarity, subjectivity, tokens = self.compute_rule_polarity(text)

        # Combine ML probabilities with rule-based polarity for fine tuning
        pos_p = prob_dict.get('positive', 0.0)
        neu_p = prob_dict.get('neutral', 0.0)
        neg_p = prob_dict.get('negative', 0.0)

        # Polarity adjustment
        if polarity > 0.3:
            pos_p += 0.25 * polarity
            neg_p -= 0.15 * polarity
        elif polarity < -0.3:
            neg_p += 0.25 * abs(polarity)
            pos_p -= 0.15 * abs(polarity)

        # Normalize
        total_p = max(pos_p + neu_p + neg_p, 1e-6)
        pos_p, neu_p, neg_p = pos_p / total_p, neu_p / total_p, neg_p / total_p

        # Primary label selection
        prob_map = {'positive': pos_p, 'neutral': neu_p, 'negative': neg_p}
        predicted_sentiment = max(prob_map, key=prob_map.get)
        confidence = float(prob_map[predicted_sentiment])

        # Enhanced token tagging using vectorizer vocabulary coefficients
        if hasattr(self.classifier, 'coef_'):
            feature_names = list(self.vectorizer.get_feature_names_out())
            feature_set = set(feature_names)
            
            for token_info in tokens:
                w = token_info["word"].lower()
                if w in feature_set and token_info["sentiment"] == "neutral":
                    idx = feature_names.index(w)
                    pos_weight = self.classifier.coef_[list(self.classifier.classes_).index('positive')][idx]
                    neg_weight = self.classifier.coef_[list(self.classifier.classes_).index('negative')][idx]

                    if pos_weight > 0.4:
                        token_info["sentiment"] = "positive"
                        token_info["score"] = round(float(pos_weight), 2)
                    elif neg_weight > 0.4:
                        token_info["sentiment"] = "negative"
                        token_info["score"] = round(float(-neg_weight), 2)

        return {
            "text": text,
            "sentiment": predicted_sentiment,
            "confidence": round(confidence * 100, 1),
            "polarity": round(polarity, 2),
            "subjectivity": round(subjectivity, 2),
            "probabilities": {
                "positive": round(pos_p * 100, 1),
                "neutral": round(neu_p * 100, 1),
                "negative": round(neg_p * 100, 1)
            },
            "tokens": tokens
        }


# Global engine instance
engine = SentimentEngine()


if __name__ == "__main__":
    test_samples = [
        "This product is amazing! Fantastic build quality.",
        "The item is okay, basic functions work.",
        "Not bad at all, actually quite impressive performance.",
        "Worst purchase ever. Completely broken and useless."
    ]
    for sample in test_samples:
        res = engine.analyze_text(sample)
        print(f"Text: '{sample}' -> Sentiment: {res['sentiment'].upper()} ({res['confidence']}%) | Polarity: {res['polarity']}")
