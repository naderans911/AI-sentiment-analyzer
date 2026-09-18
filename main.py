"""
Simple Sentiment Analyzer
A small educational NLP project using Python and VADER.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Return the sentiment label and compound score for a piece of text."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, score


def main() -> None:
    print("=" * 45)
    print("        Simple AI Sentiment Analyzer")
    print("=" * 45)
    print("Type a sentence and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() == "quit":
            print("Goodbye!")
            break

        if not text:
            print("Please enter some text.\n")
            continue

        label, score = analyze_sentiment(text)
        print(f"Sentiment: {label}")
        print(f"Confidence score: {score:.3f}\n")


if __name__ == "__main__":
    main()
