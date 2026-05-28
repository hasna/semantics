"""Seed concepts for codebook training.

Combines multiple sources:
1. Wierzbicka's 65 semantic primes (universal atoms of meaning)
2. Plutchik's emotion wheel (8 primary + blends)
3. Common speech acts / intents
4. Common topics and domains
5. Expanded phrases for richer embedding coverage
"""

# Wierzbicka's 65 semantic primes — proven universal across 36 languages
SEMANTIC_PRIMES = [
    # Substantives
    "I", "you", "someone", "something", "people", "body",
    # Determiners
    "this", "the same", "other",
    # Quantifiers
    "one", "two", "some", "all", "much", "many",
    # Evaluators
    "good", "bad",
    # Descriptors
    "big", "small",
    # Mental predicates
    "think", "know", "want", "don't want", "feel", "see", "hear",
    # Speech
    "say", "words", "true",
    # Actions/Events
    "do", "happen", "move",
    # Existence/Possession
    "there is", "be (somewhere)", "have", "be (someone/something)",
    # Life/Death
    "live", "die",
    # Time
    "when", "now", "before", "after", "a long time", "a short time", "for some time", "moment",
    # Space
    "where", "here", "above", "below", "far", "near", "side", "inside", "touch",
    # Logical
    "not", "maybe", "can", "because", "if",
    # Augmentor/Intensifier
    "very", "more",
    # Similarity
    "like (similar)",
]

# Plutchik's emotion wheel — 8 primary emotions + blends
EMOTIONS = [
    # Primary emotions
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
    # Primary dyads (blends of adjacent)
    "love", "submission", "awe", "disapproval", "remorse", "contempt", "aggressiveness", "optimism",
    # Secondary dyads
    "guilt", "delight", "curiosity", "despair", "envy", "cynicism", "pride", "hope",
    # Tertiary dyads
    "anxiety", "shame", "morbidness", "dominance", "sentimentality", "outrage", "bittersweetness", "fatalism",
    # Common emotional states
    "happiness", "frustration", "excitement", "boredom", "loneliness", "gratitude",
    "empathy", "nostalgia", "confusion", "determination", "relief", "jealousy",
    "embarrassment", "contentment", "grief", "enthusiasm", "serenity", "irritation",
]

# Speech acts / intents
INTENTS = [
    "request for help", "asking a question", "making a statement", "giving a command",
    "expressing agreement", "expressing disagreement", "offering an opinion",
    "making a suggestion", "apologizing", "thanking", "greeting", "saying goodbye",
    "warning", "promising", "threatening", "complaining", "praising", "criticizing",
    "explaining", "describing", "narrating", "arguing", "persuading", "negotiating",
    "requesting information", "confirming", "denying", "clarifying", "summarizing",
    "requesting permission", "giving permission", "refusing", "accepting", "rejecting",
    "inviting", "congratulating", "expressing sympathy", "expressing surprise",
    "expressing doubt", "expressing certainty", "expressing preference",
    "making a comparison", "making a prediction", "reporting", "instructing",
]

# Common topics/domains
TOPICS = [
    # Technology
    "programming", "software development", "debugging code", "database query",
    "web development", "mobile app", "API design", "cloud computing",
    "machine learning", "artificial intelligence", "data science", "cybersecurity",
    "operating system", "network architecture", "version control",
    # Science
    "physics", "chemistry", "biology", "mathematics", "astronomy",
    "climate science", "genetics", "neuroscience", "ecology",
    # Business
    "marketing", "finance", "management", "entrepreneurship", "sales",
    "customer service", "project management", "strategy", "accounting",
    # Health
    "medicine", "mental health", "nutrition", "exercise", "disease",
    "treatment", "diagnosis", "wellness", "healthcare",
    # Education
    "teaching", "learning", "studying", "homework", "research",
    "university", "training", "curriculum",
    # Daily life
    "cooking", "travel", "shopping", "housing", "transportation",
    "weather", "family", "friendship", "relationship", "work",
    # Creative
    "writing", "music", "art", "design", "photography",
    "film", "literature", "architecture", "fashion",
    # Society
    "politics", "law", "economics", "philosophy", "history",
    "culture", "religion", "ethics", "environment", "justice",
]

# Expanded phrases for richer embeddings (these help the model understand context)
EXPANDED_PHRASES = [
    # Requests at different formality levels
    "Can you help me with this?",
    "I need assistance with something",
    "Please fix this immediately",
    "Would you mind taking a look?",
    "Help!",
    # Technical scenarios
    "The function is throwing an error",
    "The database query is too slow",
    "The API returns a 500 error",
    "Memory usage is increasing over time",
    "The test suite is failing",
    "I need to refactor this code",
    "The deployment pipeline is broken",
    # Emotional expressions
    "I'm really frustrated with this",
    "This is amazing!",
    "I'm confused about how this works",
    "I'm worried about the deadline",
    "I feel great about the progress",
    "This is unacceptable",
    "I appreciate your help",
    # Knowledge / information
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Explain quantum computing",
    "What are the symptoms of diabetes?",
    "How do I calculate compound interest?",
    # Instructions
    "First, open the terminal",
    "Step one: install the dependencies",
    "Click the button on the top right",
    "Run the following command",
    "Navigate to the settings page",
    # Opinions and evaluations
    "I think this approach is better",
    "The quality could be improved",
    "This is the best solution I've seen",
    "I disagree with that assessment",
    "The performance is acceptable",
    # Spatial/physical
    "It's located on the left side",
    "The object is above the table",
    "Move it to the right",
    "It's far away from here",
    "The temperature is very high",
    # Temporal
    "This happened yesterday",
    "It will be ready by tomorrow",
    "The meeting is in two hours",
    "It's been running for three days",
    "The deadline was last week",
    # Quantitative
    "There are approximately 500 items",
    "The cost is $29.99",
    "It decreased by 15 percent",
    "More than half of the users",
    "The smallest possible value",
    # Logical/conditional
    "If this fails, try the alternative",
    "The error occurs only when both conditions are true",
    "Either option A or option B will work",
    "This is not what I expected",
    "It might work but I'm not sure",
    # Meta / self-referential
    "I don't understand what you mean",
    "Can you repeat that?",
    "Let me think about this",
    "That's a good point",
    "I was wrong about that",
]


def get_all_concepts() -> list[str]:
    """Return all seed concepts as a flat list."""
    all_concepts = []
    all_concepts.extend(SEMANTIC_PRIMES)
    all_concepts.extend(EMOTIONS)
    all_concepts.extend(INTENTS)
    all_concepts.extend(TOPICS)
    all_concepts.extend(EXPANDED_PHRASES)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in all_concepts:
        c_lower = c.lower().strip()
        if c_lower not in seen:
            seen.add(c_lower)
            unique.append(c)
    return unique
