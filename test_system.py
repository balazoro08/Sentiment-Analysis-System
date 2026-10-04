import unittest
from sentiment_engine import engine
from dataset_generator import get_combined_dataset

class TestSentimentAnalysisSystem(unittest.TestCase):

    def test_positive_sentiment(self):
        text = "This wireless headset is incredible! Outstanding sound clarity and fast shipping."
        res = engine.analyze_text(text)
        self.assertEqual(res["sentiment"], "positive")
        self.assertGreater(res["confidence"], 60.0)
        self.assertGreater(res["polarity"], 0.2)

    def test_negative_sentiment(self):
        text = "Terrible product! Stopped working after 2 days, complete waste of money."
        res = engine.analyze_text(text)
        self.assertEqual(res["sentiment"], "negative")
        self.assertGreater(res["confidence"], 60.0)
        self.assertLess(res["polarity"], -0.2)

    def test_neutral_sentiment(self):
        text = "The item arrived today in standard packaging. It works fine as described."
        res = engine.analyze_text(text)
        self.assertEqual(res["sentiment"], "neutral")

    def test_negation_handling(self):
        # "not bad" should be positive/neutral, NOT negative
        text = "Not bad at all, actually it is surprisingly good performance."
        res = engine.analyze_text(text)
        self.assertIn(res["sentiment"], ["positive", "neutral"])
        
        # "not good" should be negative
        text2 = "Not good. The camera quality is very poor."
        res2 = engine.analyze_text(text2)
        self.assertEqual(res2["sentiment"], "negative")

    def test_token_driver_highlighting(self):
        text = "Amazing speed but horrible battery life."
        res = engine.analyze_text(text)
        tokens = res["tokens"]
        self.assertTrue(any(t["word"].lower() == "amazing" and t["sentiment"] == "positive" for t in tokens))
        self.assertTrue(any(t["word"].lower() == "horrible" and t["sentiment"] == "negative" for t in tokens))

    def test_model_metrics(self):
        metrics = engine.model_metrics
        self.assertGreater(metrics["accuracy"], 80.0)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreater(len(metrics["top_positive_features"]), 0)


if __name__ == "__main__":
    unittest.main()
