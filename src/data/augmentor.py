"""Text augmentation for classification.

Augmentation artificially expands a training dataset by creating slightly
modified copies of existing samples.  This is especially useful when you
have **imbalanced classes** — a minority class with few examples can be
oversampled via augmentation so the model sees more variety during training.

The four classic EDA (Easy Data Augmentation) operations from Wei & Zou
(2019) are:

1. **Synonym Replacement** — swap a word for a WordNet synonym.
2. **Random Insertion** — drop a random synonym into a random position.
3. **Random Swap** — exchange the positions of two words.
4. **Random Deletion** — remove each word independently with probability *p*.

References:
    Wei, J., & Zou, K. (2019). *EDA: Easy Data Augmentation for Text
    Classification*. EMNLP-IJCNLP 2019.
"""

import logging
import random
from typing import List, Optional

import nltk
from nltk.corpus import wordnet
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure WordNet data is available — WordNet is a lexical database of
# English that groups words into sets of cognitive synonyms (synsets).
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def get_synonyms(word: str) -> List[str]:
    """Look up synonyms for *word* using WordNet.

    WordNet organises words into **synsets** (sets of synonymous words
    that express the same concept).  Each synset contains **lemmas**
    (specific word forms).  We collect all lemmas across every synset
    that contains *word*, excluding the word itself.

    Args:
        word: A single English word (case-sensitive lookup, but we
            compare lowercase to avoid duplicates like "Run" vs "run").

    Returns:
        A list of unique synonym strings.  May be empty if WordNet has
        no synonyms for the given word.

    Examples:
        >>> get_synonyms("happy")
        ['felicitous', 'glad', 'joyous']
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Exclude the original word so we don't "replace" it with itself
            if lemma.name().lower() != word.lower():
                # WordNet uses underscores for multi-word expressions (e.g. "good_morning")
                synonyms.add(lemma.name().lower().replace("_", " "))
    return list(synonyms)


def synonym_replacement(text: str, n: int = 1, seed: int = 42) -> str:
    """Replace *n* randomly chosen words with WordNet synonyms.

    This is the simplest augmentation: pick words at random and swap each
    one for a synonym.  The meaning of the sentence is preserved while
    the model sees a different surface form — forcing it to generalise
    beyond exact vocabulary.

    Args:
        text: Input sentence or paragraph.
        n: Number of words to attempt to replace.  If *n* exceeds the
            number of words in *text*, all words are considered.
        seed: Random seed for reproducibility.

    Returns:
        Augmented text with up to *n* words replaced.  Short texts
        (fewer than 4 words) are returned unchanged to avoid losing
        too much semantic content.

    Examples:
        >>> synonym_replacement("The quick brown fox jumps", n=2, seed=0)
        'The quick brown fox jumps'  # depends on synonyms available
    """
    random.seed(seed)
    words = text.split()

    # Guard rail: very short texts lose too much meaning if we replace words
    if len(words) < 4:
        return text

    new_words = words.copy()
    # min() prevents sampling more indices than exist
    indices = random.sample(range(len(words)), min(n, len(words)))

    for idx in indices:
        syns = get_synonyms(words[idx])
        # Only replace if synonyms exist — some words (proper nouns, slang)
        # have no WordNet entry
        if syns:
            new_words[idx] = random.choice(syns)

    return " ".join(new_words)


def random_deletion(text: str, p: float = 0.1, seed: int = 42) -> str:
    """Randomly delete words with probability *p*.

    Each word is independently removed with probability *p*.  This
    teaches the model to be robust to missing or noisy input — a
    real-world scenario when OCR or ASR introduces dropouts.

    A low probability (e.g. 0.05–0.15) is recommended; higher values
    can destroy sentence coherence.

    Args:
        text: Input sentence or paragraph.
        p: Probability of deleting any single word.  Must be in [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        Text with words randomly removed.  Returns unchanged if the text
        has 2 or fewer words (we always keep at least the subject).

    Examples:
        >>> random_deletion("the cat sat on the mat", p=0.3, seed=0)
        'the cat on the'
    """
    random.seed(seed)
    words = text.split()

    # Never delete from very short texts — we need at least some context
    if len(words) <= 2:
        return text

    return " ".join(w for w in words if random.random() > p)


def random_swap(text: str, n: int = 1, seed: int = 42) -> str:
    """Randomly swap *n* pairs of adjacent or non-adjacent words.

    Swapping word order forces the model to rely on distributional
    semantics rather than positional heuristics.  For example, in
    sentiment analysis, "good not bad" vs "bad not good" should both
    be recognised as positive.

    Args:
        text: Input sentence or paragraph.
        n: Number of swap operations to perform.  Each swap picks
            two random positions (not necessarily adjacent).
        seed: Random seed for reproducibility.

    Returns:
        Text with *n* random word swaps applied.

    Examples:
        >>> random_swap("the cat sat on the mat", n=1, seed=42)
        'the mat sat on the cat'
    """
    random.seed(seed)
    words = text.split()

    if len(words) < 2:
        return text

    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]

    return " ".join(words)


def random_insertion(text: str, n: int = 1, seed: int = 42) -> str:
    """Insert *n* random synonyms at random positions in the text.

    This augmentation **adds** information rather than removing or
    substituting it.  For each insertion we:

    1. Pick a random word from the sentence.
    2. Find a synonym for it via WordNet.
    3. Insert that synonym at a random position.

    The result is a slightly longer sentence that preserves the
    original meaning — useful for making the model tolerant of
    wordy or verbose inputs.

    Args:
        text: Input sentence or paragraph.
        n: Number of synonym insertions to perform.
        seed: Random seed for reproducibility.

    Returns:
        Text with *n* additional synonym words inserted.  Returns
        unchanged if the text has fewer than 2 words.

    Examples:
        >>> random_insertion("the cat sat", n=1, seed=42)
        'the cat quick sat'  # "quick" inserted as synonym of some word
    """
    random.seed(seed)
    words = text.split()

    # Need at least 2 words: one to source a synonym from, and a position
    if len(words) < 2:
        return text

    for _ in range(n):
        # Step 1: pick a random word from the existing sentence
        source_idx = random.choice(range(len(words)))
        syns = get_synonyms(words[source_idx])

        # Step 2: only proceed if we found a synonym
        if syns:
            synonym = random.choice(syns)
            # Step 3: insert at a random position (can be before, after,
            # or between any two existing words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, synonym)

    return " ".join(words)


def augment_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    target_column: Optional[str] = None,
    augment_fn=None,
    augment_factor: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance a dataset by augmenting minority-class samples.

    In imbalanced classification, the majority class dominates the
    loss gradient and the model learns to ignore minority classes.
    Oversampling via text augmentation addresses this without
    discarding majority-class data (as undersampling would).

    For each minority class, this function duplicates existing
    samples and applies *augment_fn* to create new variations until
    the class count matches the majority class.

    Args:
        df: Input DataFrame containing text samples and labels.
        text_column: Name of the column containing the raw text.
        label_column: Name of the column containing class labels.
        target_column: Name of the column that will hold augmented
            text in the output.  If ``None``, the original *text_column*
            is overwritten with augmented text for new rows.
        augment_fn: A callable ``(text, seed) -> str`` that performs
            augmentation on a single text string.  Defaults to
            :func:`synonym_replacement` with one replacement.
        augment_factor: How many augmented copies to create per
            original minority sample per round.  Higher values reach
            balance faster but may produce lower-quality samples.
        seed: Base random seed.  Each sample gets ``seed + i`` to
            ensure diverse but reproducible augmentations.

    Returns:
        A new DataFrame with the original rows plus augmented minority-
        class rows.  The original DataFrame is **not** modified.

    Raises:
        ValueError: If *text_column* or *label_column* is not in *df*.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "text": ["good movie", "bad film", "terrible plot"],
        ...     "label": ["pos", "neg", "neg"],
        ... })
        >>> balanced = augment_dataset(df, augment_fn=synonym_replacement)
        >>> balanced["label"].value_counts()
        neg    3
        pos    2
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")

    # Default augmentation function if none provided
    if augment_fn is None:
        augment_fn = synonym_replacement

    # Find the majority class count — our target for all classes
    class_counts = df[label_column].value_counts()
    majority_count = class_counts.max()

    augmented_rows: List[dict] = []

    for label, count in class_counts.items():
        if count >= majority_count:
            # This class is already at or above the target — skip it
            continue

        # Number of new samples needed to reach parity with the majority
        samples_needed = majority_count - count

        # Filter to this class's rows
        class_df = df[df[label_column] == label]

        for i in range(samples_needed):
            # Cycle through existing samples — wrapping with modulo
            original_row = class_df.iloc[i % len(class_df)]
            original_text = original_row[text_column]

            # Each sample gets a unique seed for diverse augmentations
            augmented_text = augment_fn(original_text, seed=seed + i)

            new_row = original_row.to_dict()
            out_col = target_column if target_column else text_column
            new_row[out_col] = augmented_text
            augmented_rows.append(new_row)

    # Combine originals with augmented rows
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        return pd.concat([df, augmented_df], ignore_index=True)

    return df.copy()
