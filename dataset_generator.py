import pandas as pd
import random
import os

# Comprehensive training data for Sentiment Analysis across domains
PRODUCT_REVIEWS = [
    # Positive
    ("This product exceeded all my expectations! Outstanding quality and build.", "positive"),
    ("Absolutely fantastic! Works seamlessly right out of the box.", "positive"),
    ("Best purchase I have made this year. High performance and durable.", "positive"),
    ("Super fast shipping, wonderful customer support, and amazing value for money.", "positive"),
    ("I love everything about this gadget. Highly recommended to everyone!", "positive"),
    ("Sleek design, top-tier performance, and battery life is incredible.", "positive"),
    ("Great value! Does the job perfectly without any hassle.", "positive"),
    ("Impressed with the craftsmanship. Truly premium quality.", "positive"),
    ("Five stars! Will definitely buy again from this brand.", "positive"),
    ("Not bad at all, actually it is surprisingly excellent!", "positive"),
    ("Never fails to impress me. Constant reliability.", "positive"),
    ("Easy to setup, elegant interface, and very user friendly.", "positive"),
    ("Works like a charm! A complete game changer for my daily routine.", "positive"),
    ("Extremely satisfied. Worth every single penny.", "positive"),
    ("Solid build quality, crystal clear audio, and fast charging.", "positive"),

    # Neutral
    ("The item arrived today. Package was intact and contents match description.", "neutral"),
    ("Average performance. It is neither great nor terrible, just okay.", "neutral"),
    ("Functions as expected according to the manual specifications.", "neutral"),
    ("Decent product for the price point, but nothing extraordinary.", "neutral"),
    ("It works fine. Standard build quality with basic features.", "neutral"),
    ("Received the package in 3 days. Standard packaging.", "neutral"),
    ("Fair product. Does what it says, no more no less.", "neutral"),
    ("The color is slightly lighter than shown in pictures, but acceptable.", "neutral"),
    ("It performs basic tasks adequately.", "neutral"),
    ("Middle-of-the-road quality. Acceptable for occasional use.", "neutral"),
    ("Standard functionality. Meets basic requirements.", "neutral"),
    ("It is okay. Nothing special to write home about.", "neutral"),

    # Negative
    ("Terrible quality! Stopped working after just two days.", "negative"),
    ("Complete waste of money. Do not buy this item!", "negative"),
    ("Horrible customer service and defective product delivered.", "negative"),
    ("Extremely disappointing. The material feels cheap and flimsy.", "negative"),
    ("Overpriced junk. Failed to live up to any promises.", "negative"),
    ("Very poor performance. Constant glitches and overheating.", "negative"),
    ("Not good. Battery dies in less than an hour of usage.", "negative"),
    ("Arrived broken and customer support refuses to process a refund.", "negative"),
    ("I regret buying this. Worst experience ever.", "negative"),
    ("Extremely slow delivery, missing parts, and damaged box.", "negative"),
    ("Fails completely at its core task. Very frustrating.", "negative"),
    ("Not worth the money. Cheap plastic and broken buttons.", "negative"),
    ("Don't buy! Horrible design and totally useless.", "negative"),
    ("Defective unit. Loud noise, poor quality control.", "negative")
]

SOCIAL_MEDIA_COMMENTS = [
    # Positive
    ("Loving the new update! Smooth UI and awesome new features 🔥", "positive"),
    ("Huge shoutout to the dev team! This project is inspiring ❤️", "positive"),
    ("Just tried this out and I'm blown away by how fast it is ✨", "positive"),
    ("Can't stop using this app! Absolutely brilliant implementation.", "positive"),
    ("Super excited for the upcoming features! Keep up the great work 🎉", "positive"),
    ("This is revolutionary! Best news I've heard all week 🚀", "positive"),
    ("Pure perfection! Highly recommended to all my friends.", "positive"),
    ("So helpful and intuitive. Thank you for building this!", "positive"),
    ("Great community and awesome support. Love it!", "positive"),
    ("Not half bad! Actually turned out to be amazing.", "positive"),

    # Neutral
    ("Just posted a new update. Check out the link in bio.", "neutral"),
    ("The event starts at 5 PM PST tomorrow.", "neutral"),
    ("Looking forward to seeing how this evolves over time.", "neutral"),
    ("The server maintenance is scheduled for tonight at midnight.", "neutral"),
    ("Here are the stats for this month's activity.", "neutral"),
    ("Interesting concept. Let's see how it performs in production.", "neutral"),
    ("Updated the repository with the latest documentation.", "neutral"),
    ("The webinar recording is now available on YouTube.", "neutral"),
    ("New version released today. Release notes available online.", "neutral"),
    ("A standard update containing routine bug fixes.", "neutral"),

    # Negative
    ("Another broken update... nothing works anymore 😡", "negative"),
    ("Worst user experience ever. Frustrating navigation and constant crashes.", "negative"),
    ("Extremely disappointed with the recent policy changes.", "negative"),
    ("Total nightmare to setup. Unhelpful documentation.", "negative"),
    ("Buggy software, unresponsive support, completely useless.", "negative"),
    ("Why did you remove the best feature? Ruined the app.", "negative"),
    ("Waste of time! Don't waste your energy on this project.", "negative"),
    ("Constant downtime and lost data. Avoid at all costs!", "negative"),
    ("Terrible performance on mobile devices. Unusable.", "negative"),
    ("Not happy with this service at all. Canceling my subscription.", "negative")
]

APP_REVIEWS = [
    # Positive
    ("Best productivity app on the store! Seamless sync across devices.", "positive"),
    ("Intuitive UI, clean workflow, and zero lag. 5 stars!", "positive"),
    ("Saved me hours of work every week. Indispensable tool!", "positive"),
    ("Superb battery optimization and frequent quality updates.", "positive"),
    ("Flawless execution! Highly polished interface.", "positive"),
    
    # Neutral
    ("Does what it promises. UI could be modernized, but functional.", "neutral"),
    ("Basic features work as expected. Premium tier is a bit pricey.", "neutral"),
    ("App size is large, but features are standard.", "neutral"),
    ("Average app. Good for casual users.", "neutral"),
    
    # Negative
    ("App crashes every time I try to open a document.", "negative"),
    ("Full of intrusive ads and paywalls everywhere. Uninstalled.", "negative"),
    ("Drains battery insanely fast and freezes constantly.", "negative"),
    ("Lost all my saved data after the latest patch! Horrible.", "negative")
]


def get_combined_dataset():
    """Returns combined dataset as pandas DataFrame."""
    all_data = PRODUCT_REVIEWS + SOCIAL_MEDIA_COMMENTS + APP_REVIEWS
    # Duplicate and expand slightly with variations for robust training
    expanded = []
    for text, label in all_data:
        expanded.append({'text': text, 'sentiment': label})
    
    return pd.DataFrame(expanded)


def generate_sample_csv(file_path="sample_reviews.csv"):
    """Generates a sample CSV file with mixed reviews for batch analysis demonstration."""
    sample_items = [
        {"id": 101, "domain": "Product Review", "text": "This wireless headset is incredible! The sound clarity and active noise cancellation are world-class."},
        {"id": 102, "domain": "Product Review", "text": "Battery life is mediocre, lasting around 4 hours. It is acceptable for commuting."},
        {"id": 103, "domain": "Product Review", "text": "Arrived severely scratched and right earbud doesn't charge. Extremely unsatisfied."},
        {"id": 104, "domain": "Social Media", "text": "Just tried the new platform update and the UI design is sleek and responsive! 👏"},
        {"id": 105, "domain": "Social Media", "text": "The scheduled maintenance announcement was posted earlier today."},
        {"id": 106, "domain": "Social Media", "text": "Worst update ever! The app crashes repeatedly and logs me out continuously 😡"},
        {"id": 107, "domain": "Customer Support", "text": "Agent Sarah was super helpful, patient, and solved my billing issue in 5 minutes. Excellent service!"},
        {"id": 108, "domain": "Customer Support", "text": "My ticket #8921 is still open after 4 business days. No status updates received."},
        {"id": 109, "domain": "Customer Support", "text": "Support requested my account details for further verification."},
        {"id": 110, "domain": "App Store Review", "text": "Five stars! The offline mode feature is a lifesaver during long flights."},
        {"id": 111, "domain": "App Store Review", "text": "Too many popups asking for reviews every 2 minutes. Quite annoying."},
        {"id": 112, "domain": "App Store Review", "text": "App crashes immediately on launch after updating to iOS 18."}
    ]
    
    df = pd.DataFrame(sample_items)
    df.to_csv(file_path, index=False)
    print(f"Sample CSV generated at {file_path}")
    return file_path


if __name__ == "__main__":
    df = get_combined_dataset()
    print(f"Total training samples: {len(df)}")
    print(df['sentiment'].value_counts())
    generate_sample_csv()
