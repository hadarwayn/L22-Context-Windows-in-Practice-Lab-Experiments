"""
Document generator for Context Windows Lab experiments.

Generates synthetic documents with embedded facts at specified positions.
"""

import random
from typing import List, Tuple

# Filler text templates by topic
FILLER_TEMPLATES = {
    "business": [
        "The quarterly financial results showed significant growth across all sectors.",
        "Market analysis indicates strong demand for innovative solutions.",
        "Strategic partnerships continue to drive organizational development.",
        "Customer satisfaction metrics have improved substantially this period.",
        "Investment in research and development remains a top priority.",
        "The management team is focused on sustainable business practices.",
        "Operational efficiency has been enhanced through process optimization.",
        "Employee engagement surveys reveal positive workplace culture trends.",
    ],
    "technology": [
        "Cloud computing infrastructure enables scalable application deployment.",
        "Machine learning algorithms process large datasets efficiently.",
        "Cybersecurity measures protect sensitive data from threats.",
        "Software development follows agile methodology principles.",
        "Network architecture supports high availability requirements.",
        "Database optimization improves query performance significantly.",
        "API integration facilitates seamless system communication.",
        "Automated testing ensures code quality and reliability.",
    ],
    "neutral": [
        "The weather conditions varied throughout the observation period.",
        "Various factors contributed to the overall outcome of events.",
        "Multiple perspectives were considered during the analysis phase.",
        "Documentation of procedures ensures consistency in operations.",
        "Regular reviews maintain quality standards across departments.",
        "Communication channels facilitate information exchange effectively.",
        "Planning activities support successful project implementation.",
        "Resource allocation follows established guidelines carefully.",
    ]
}

HEBREW_TEMPLATES = {
    "technology": [
        "מערכות מידע מתקדמות מאפשרות עיבוד נתונים בקנה מידה גדול.",
        "פיתוח תוכנה מודרני מבוסס על עקרונות של קוד נקי.",
        "אבטחת מידע היא עדיפות עליונה בארגונים טכנולוגיים.",
        "בינה מלאכותית משנה את אופן העבודה בתעשייה.",
    ],
    "law": [
        "החקיקה החדשה מסדירה את תחום הטכנולוגיה הפיננסית.",
        "זכויות הפרט מוגנות על פי חוק יסוד כבוד האדם.",
        "הליכים משפטיים דורשים תיעוד מדויק של כל הפעולות.",
        "רגולציה בתחום הבריאות מבטיחה סטנדרטים גבוהים.",
    ],
    "medicine": [
        "מחקרים קליניים בוחנים יעילות של טיפולים חדשניים.",
        "התרופה X עלולה לגרום לתופעות לוואי כמו סחרחורת ובחילה.",
        "אבחון מוקדם משפר משמעותית את סיכויי ההחלמה.",
        "טכנולוגיות רפואיות מתקדמות מאפשרות ניתוחים מדויקים.",
    ]
}


def generate_filler_text(num_words: int, topic: str = "neutral") -> str:
    """
    Generate filler text of approximately specified word count.

    Args:
        num_words: Target number of words
        topic: Topic for filler text (business, technology, neutral)

    Returns:
        Generated filler text
    """
    templates = FILLER_TEMPLATES.get(topic, FILLER_TEMPLATES["neutral"])
    sentences = []
    word_count = 0

    while word_count < num_words:
        sentence = random.choice(templates)
        sentences.append(sentence)
        word_count += len(sentence.split())

    return " ".join(sentences)


def embed_fact_at_position(text: str, fact: str, position: str) -> Tuple[str, int]:
    """
    Embed a fact at a specified position in the text.

    Args:
        text: Base text to embed fact into
        fact: The fact to embed
        position: Where to embed ('start', 'middle', 'end')

    Returns:
        Tuple of (modified text, fact position index)
    """
    words = text.split()
    fact_words = fact.split()

    if position == "start":
        idx = 0
        result = fact + " " + text
    elif position == "end":
        idx = len(words)
        result = text + " " + fact
    else:  # middle
        idx = len(words) // 2
        result = " ".join(words[:idx]) + " " + fact + " " + " ".join(words[idx:])

    return result, idx


def generate_document(
    total_words: int,
    fact: str,
    position: str,
    topic: str = "neutral"
) -> str:
    """
    Generate a document with an embedded fact.

    Args:
        total_words: Target total word count
        fact: Fact to embed
        position: Position for fact ('start', 'middle', 'end')
        topic: Topic for filler text

    Returns:
        Generated document with embedded fact
    """
    fact_words = len(fact.split())
    filler_words = max(10, total_words - fact_words)
    filler = generate_filler_text(filler_words, topic)
    document, _ = embed_fact_at_position(filler, fact, position)
    return document


def generate_document_set(
    num_docs: int,
    words_per_doc: int,
    topic: str = "neutral"
) -> List[str]:
    """
    Generate a set of documents without embedded facts.

    Args:
        num_docs: Number of documents to generate
        words_per_doc: Words per document

    Returns:
        List of generated documents
    """
    return [generate_filler_text(words_per_doc, topic) for _ in range(num_docs)]


def generate_hebrew_documents(
    num_docs: int,
    topics: List[str]
) -> List[Tuple[str, str]]:
    """
    Generate Hebrew documents for RAG experiment.

    Args:
        num_docs: Number of documents
        topics: List of topics to use

    Returns:
        List of (document, topic) tuples
    """
    documents = []

    # Ensure the side effects fact is always included in one medicine document
    side_effects_fact = "התרופה X עלולה לגרום לתופעות לוואי כמו סחרחורת ובחילה."

    for i in range(num_docs):
        topic = topics[i % len(topics)]
        templates = HEBREW_TEMPLATES.get(topic, HEBREW_TEMPLATES["technology"])

        if topic == "medicine" and i < 3:
            # Ensure side effects are in early medicine documents
            doc = side_effects_fact + " " + " ".join(
                random.sample([t for t in templates if t != side_effects_fact],
                              min(len(templates) - 1, 2))
            )
        else:
            doc = " ".join(random.sample(templates, min(len(templates), 3)))

        documents.append((doc, topic))
    return documents
