"""
Perturbation Rules for Ambiguity Detection via Input Uncertainty.

Group A: Single Word Replacement (by POS)
  A1 - Verb Synonym
  A2 - Noun Synonym
  A3 - Adjective Synonym
  A4 - Adverb Synonym
  A5 - Abbreviation Expand
  A6 - Abbreviation Contract

Group B-Easy: Prefix Insertion (no sentence understanding needed)
  B1 - Test-like Prefix
  B2 - Conversation-like Prefix
  B3 - Instruction Prefix

Group B-Template: WH-word Template Replacement
  B4 - WH Expansion
  B5 - WH Rephrase
"""

import re
import random
from typing import List, Dict, Tuple, Optional

# ============================================================
# WordNet and spaCy — WordNet is REQUIRED for synonym rules
# ============================================================
try:
    from nltk.corpus import wordnet as wn
    from nltk import pos_tag, word_tokenize
    _synsets = wn.synsets  # test if data is loaded
    _synsets("test")
    USE_WORDNET = True
except Exception:
    USE_WORDNET = False
    print("WARNING: WordNet not available. Synonym rules (A1-A4) will not work.")
    print("Install with:")
    print("  pip install nltk")
    print("  python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False


# ============================================================
# Abbreviation mappings
# ============================================================
CONTRACTIONS = {
    # expand: contraction -> expanded
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
    "why's": "why is",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i'm": "i am",
    "they've": "they have",
    "we've": "we have",
    "you've": "you have",
    "i've": "i have",
    "they'll": "they will",
    "we'll": "we will",
    "you'll": "you will",
    "i'll": "i will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'd": "they would",
    "we'd": "we would",
    "you'd": "you would",
    "i'd": "i would",
    "he'd": "he would",
    "she'd": "she would",
}

# Reverse mapping for contraction
EXPANSIONS = {}
for contraction, expansion in CONTRACTIONS.items():
    EXPANSIONS[expansion] = contraction


# ============================================================
# WH-word mappings
# ============================================================
WH_EXPANSION = {
    "who": "which person",
    "what": "which thing",
    "where": "in what place",
    "when": "at what time",
    "why": "for what reason",
    "how": "in what way",
    "how many": "what number of",
    "how much": "what amount of",
    "how long": "for what duration",
    "how often": "at what frequency",
    "how old": "what age",
}

WH_REPHRASE = {
    "who": ["what person", "whom"],
    "what": ["which", "what kind of thing"],
    "where": ["in which location", "at what place"],
    "when": ["at what point", "at which time"],
    "why": ["what is the reason that", "how come"],
    "how": ["by what means", "in what manner"],
    "how many": ["what is the count of", "what quantity of"],
    "how much": ["what quantity of", "to what extent"],
    "how long": ["what is the duration", "for how much time"],
    "how often": ["how frequently", "with what frequency"],
    "how old": ["what is the age of"],
}


# ============================================================
# Prefix lists for B-Easy
# ============================================================
TEST_PREFIXES = [
    "According to the passage,",
    "Based on the text,",
    "As stated in the passage,",
    "From the passage,",
    "As mentioned in the text,",
]

CONVERSATION_PREFIXES = [
    "Hey so,",
    "OK so,",
    "Alright,",
    "So then,",
    "Right, so",
]

INSTRUCTION_PREFIXES = [
    "Based on the context above,",
    "Given the information provided,",
    "Considering the passage,",
    "Referring to the text above,",
    "With regard to the passage,",
]


# ============================================================
# POS tagging (simple regex-based fallback)
# ============================================================
def _simple_pos_tag(question: str) -> List[Tuple[str, str]]:
    """
    Simple POS tagging using spaCy if available, else regex heuristics.
    Returns list of (word, POS) where POS is one of:
    VERB, NOUN, ADJ, ADV, PRON, DET, ADP, OTHER
    """
    if USE_SPACY:
        doc = _nlp(question)
        return [(token.text, token.pos_) for token in doc]

    # Regex-based fallback
    words = question.split()
    result = []
    # Common word -> POS mappings for question words
    wh_words = {"who", "what", "where", "when", "why", "how", "which", "whom"}
    pronouns = {"he", "she", "it", "they", "we", "you", "i", "him", "her",
                "them", "us", "his", "its", "their", "our", "your", "my",
                "this", "that", "these", "those"}
    determiners = {"the", "a", "an", "some", "any", "each", "every", "all", "no"}
    prepositions = {"in", "on", "at", "to", "for", "with", "from", "by", "of",
                    "about", "into", "through", "during", "before", "after",
                    "above", "below", "between", "under", "over"}
    aux_verbs = {"is", "are", "was", "were", "do", "does", "did", "has", "have",
                 "had", "will", "would", "could", "should", "can", "may", "might",
                 "shall", "must", "been", "being", "am"}
    common_adverbs = {"finally", "also", "then", "still", "already", "often",
                      "never", "always", "quickly", "slowly", "nearly", "actually",
                      "sometimes", "recently", "only", "even", "just", "really",
                      "severely", "eventually", "probably", "usually", "certainly"}

    for w in words:
        w_lower = w.lower().rstrip("?.,!;:")
        if w_lower in wh_words:
            result.append((w, "WH"))
        elif w_lower in pronouns:
            result.append((w, "PRON"))
        elif w_lower in determiners:
            result.append((w, "DET"))
        elif w_lower in prepositions:
            result.append((w, "ADP"))
        elif w_lower in aux_verbs:
            result.append((w, "AUX"))
        elif w_lower in common_adverbs:
            result.append((w, "ADV"))
        elif w_lower.endswith("ly"):
            result.append((w, "ADV"))
        elif w_lower.endswith(("ing", "ed", "en")):
            result.append((w, "VERB"))
        elif w_lower.endswith(("tion", "ment", "ness", "ity", "ance", "ence")):
            result.append((w, "NOUN"))
        elif w_lower.endswith(("ful", "ous", "ive", "able", "ible", "al", "ical")):
            result.append((w, "ADJ"))
        else:
            result.append((w, "NOUN"))  # default to NOUN for content words
    return result


# ============================================================
# Morphology: inflect synonyms to match original word form
# ============================================================

# Try lemminflect first (best quality), fall back to regex rules
try:
    import lemminflect
    USE_LEMMINFLECT = True
except ImportError:
    USE_LEMMINFLECT = False


def _get_lemma(word: str, pos: str) -> str:
    """Get the base/lemma form of a word."""
    if USE_WORDNET:
        from nltk.stem import WordNetLemmatizer
        pos_map = {"VERB": "v", "NOUN": "n", "ADJ": "a", "ADV": "r"}
        wnl = WordNetLemmatizer()
        wn_pos = pos_map.get(pos, "n")
        return wnl.lemmatize(word.lower(), wn_pos)
    return word.lower()


def _detect_verb_form(word: str, lemma: str) -> str:
    """
    Detect the inflection form of a verb.
    Returns one of: 'base', 'past', 'past_participle', 'present_participle',
                     'third_person', 'unknown'
    """
    w = word.lower()
    if w == lemma:
        return "base"
    if w.endswith("ing"):
        return "present_participle"
    if w.endswith("ed"):
        return "past"
    # Irregular past tense detection: if word != lemma and doesn't end in
    # -ing or -ed, it's likely irregular past
    if w != lemma:
        return "past"
    return "base"


def _detect_noun_form(word: str, lemma: str) -> str:
    """Detect if noun is singular or plural."""
    w = word.lower()
    if w == lemma:
        return "singular"
    return "plural"


def _detect_adj_form(word: str, lemma: str) -> str:
    """Detect adjective form: base, comparative, superlative."""
    w = word.lower()
    if w.endswith("est"):
        return "superlative"
    if w.endswith("er") and len(w) > 3:
        return "comparative"
    return "base"


def _inflect_verb(lemma: str, form: str) -> Optional[str]:
    """Inflect a verb lemma to the target form."""
    if USE_LEMMINFLECT:
        import lemminflect as li
        tag_map = {
            "base": "VB",
            "past": "VBD",
            "past_participle": "VBN",
            "present_participle": "VBG",
            "third_person": "VBZ",
        }
        tag = tag_map.get(form, "VB")
        results = li.getInflection(lemma, tag=tag)
        if results:
            return results[0]

    # Regex-based fallback
    if form == "base":
        return lemma
    elif form == "past":
        if lemma.endswith("e"):
            return lemma + "d"
        elif lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ied"
        elif (len(lemma) > 2 and lemma[-1] not in "aeiouwxy"
              and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou"):
            return lemma + lemma[-1] + "ed"
        else:
            return lemma + "ed"
    elif form == "present_participle":
        if lemma.endswith("ie"):
            return lemma[:-2] + "ying"
        elif lemma.endswith("e") and not lemma.endswith("ee"):
            return lemma[:-1] + "ing"
        elif (len(lemma) > 2 and lemma[-1] not in "aeiouwxy"
              and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou"):
            return lemma + lemma[-1] + "ing"
        else:
            return lemma + "ing"
    elif form == "past_participle":
        # Same as past for regular verbs
        return _inflect_verb(lemma, "past")
    elif form == "third_person":
        if lemma.endswith(("s", "sh", "ch", "x", "z")):
            return lemma + "es"
        elif lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ies"
        else:
            return lemma + "s"
    return lemma


def _inflect_noun(lemma: str, form: str) -> Optional[str]:
    """Inflect a noun lemma to singular/plural."""
    if form == "singular":
        return lemma

    if USE_LEMMINFLECT:
        import lemminflect as li
        results = li.getInflection(lemma, tag="NNS")
        if results:
            return results[0]

    # Regex-based plural
    if lemma.endswith(("s", "sh", "ch", "x", "z")):
        return lemma + "es"
    elif lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ies"
    elif lemma.endswith("f"):
        return lemma[:-1] + "ves"
    elif lemma.endswith("fe"):
        return lemma[:-2] + "ves"
    else:
        return lemma + "s"


def _inflect_adj(lemma: str, form: str) -> Optional[str]:
    """Inflect adjective to comparative/superlative."""
    if form == "base":
        return lemma

    if USE_LEMMINFLECT:
        import lemminflect as li
        tag = "JJR" if form == "comparative" else "JJS"
        results = li.getInflection(lemma, tag=tag)
        if results:
            return results[0]

    # Regex-based
    if form == "comparative":
        if lemma.endswith("e"):
            return lemma + "r"
        elif lemma.endswith("y") and len(lemma) > 1:
            return lemma[:-1] + "ier"
        else:
            return lemma + "er"
    elif form == "superlative":
        if lemma.endswith("e"):
            return lemma + "st"
        elif lemma.endswith("y") and len(lemma) > 1:
            return lemma[:-1] + "iest"
        else:
            return lemma + "est"
    return lemma


def _inflect_synonym(synonym_lemma: str, original_word: str,
                     original_lemma: str, pos: str) -> Optional[str]:
    """
    Inflect a synonym lemma to match the morphological form of the original word.
    E.g., original="helped" (past tense of "help") + synonym_lemma="assist"
          -> returns "assisted"
    """
    if pos == "VERB":
        form = _detect_verb_form(original_word, original_lemma)
        return _inflect_verb(synonym_lemma, form)
    elif pos == "NOUN":
        form = _detect_noun_form(original_word, original_lemma)
        return _inflect_noun(synonym_lemma, form)
    elif pos == "ADJ":
        form = _detect_adj_form(original_word, original_lemma)
        return _inflect_adj(synonym_lemma, form)
    elif pos == "ADV":
        # Adverbs don't inflect in English
        return synonym_lemma
    return synonym_lemma


def _get_synonyms_wordnet(word: str, pos_tag: str) -> List[str]:
    """
    Get synonyms from WordNet for a given word and POS.
    Returns synonyms INFLECTED to match the original word's form.

    Quality filters:
    - Only use the FIRST synset (most common meaning)
    - Skip multi-word synonyms
    - Skip very short or very long synonyms
    - Skip rare words (frequency count == 0 in WordNet corpus)
    """
    pos_map = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}
    wn_pos = pos_map.get(pos_tag)
    if wn_pos is None:
        return []

    # Get lemma of the original word for WordNet lookup
    original_lemma = _get_lemma(word, pos_tag)

    synsets = wn.synsets(original_lemma, pos=wn_pos)
    if not synsets:
        # Try with original word form if lemma lookup fails
        synsets = wn.synsets(word, pos=wn_pos)
        if not synsets:
            return []

    # Only use the first synset (most common sense)
    first_synset = synsets[0]

    synonyms = []
    for lemma in first_synset.lemmas():
        syn_lemma = lemma.name().replace("_", " ")
        syn_lower = syn_lemma.lower()

        # Skip if same as original lemma
        if syn_lower == original_lemma.lower():
            continue
        # Skip multi-word synonyms
        if " " in syn_lemma:
            continue
        # Skip very short synonyms
        if len(syn_lemma) <= 2:
            continue
        # Skip if length ratio is too extreme
        if len(original_lemma) > 0 and (
            len(syn_lemma) / len(original_lemma) > 3.0
            or len(syn_lemma) / len(original_lemma) < 0.3
        ):
            continue
        # Skip rare words (frequency count 0)
        if lemma.count() == 0:
            continue

        # Inflect synonym to match original word form
        inflected = _inflect_synonym(syn_lower, word, original_lemma, pos_tag)
        if inflected is None:
            continue

        synonyms.append((inflected, lemma.count()))

    # Sort by frequency (most common first), return just the names
    synonyms.sort(key=lambda x: -x[1])
    return [s[0] for s in synonyms]


def _get_synonyms(word: str, pos: str) -> List[str]:
    """Get synonyms from WordNet only. Returns empty list if WordNet unavailable."""
    if not USE_WORDNET:
        return []
    word_lower = word.lower().rstrip("?.,!;:")
    return _get_synonyms_wordnet(word_lower, pos)


# ============================================================
# GROUP A: Single Word Replacement Rules
# ============================================================

def apply_synonym_by_pos(question: str, target_pos: str, max_variants: int = 3) -> List[Dict]:
    """
    Replace words of a specific POS with synonyms.
    Returns list of {variant_question, replaced_word, replacement, rule_id}
    """
    tagged = _simple_pos_tag(question)
    variants = []

    for i, (word, pos) in enumerate(tagged):
        if pos != target_pos:
            continue
        word_clean = word.rstrip("?.,!;:")
        trailing = word[len(word_clean):]  # preserve punctuation
        synonyms = _get_synonyms(word_clean, target_pos)
        if not synonyms:
            continue

        selected = synonyms[:max_variants]
        for syn in selected:
            # Preserve original capitalization
            if word_clean[0].isupper():
                syn = syn.capitalize()
            new_words = [t[0] for t in tagged]
            new_words[i] = syn + trailing
            new_q = " ".join(new_words)
            variants.append({
                "variant_question": new_q,
                "replaced_word": word_clean,
                "replacement": syn,
            })

    return variants


def A1_verb_synonym(question: str, max_variants: int = 3) -> List[Dict]:
    """A1: Replace verbs with synonyms."""
    results = apply_synonym_by_pos(question, "VERB", max_variants)
    for r in results:
        r["rule_id"] = "A1"
        r["rule_name"] = "Verb Synonym"
    return results


def A2_noun_synonym(question: str, max_variants: int = 3) -> List[Dict]:
    """A2: Replace nouns with synonyms."""
    results = apply_synonym_by_pos(question, "NOUN", max_variants)
    for r in results:
        r["rule_id"] = "A2"
        r["rule_name"] = "Noun Synonym"
    return results


def A3_adj_synonym(question: str, max_variants: int = 3) -> List[Dict]:
    """A3: Replace adjectives with synonyms."""
    results = apply_synonym_by_pos(question, "ADJ", max_variants)
    for r in results:
        r["rule_id"] = "A3"
        r["rule_name"] = "Adjective Synonym"
    return results


def A4_adv_synonym(question: str, max_variants: int = 3) -> List[Dict]:
    """A4: Replace adverbs with synonyms."""
    results = apply_synonym_by_pos(question, "ADV", max_variants)
    for r in results:
        r["rule_id"] = "A4"
        r["rule_name"] = "Adverb Synonym"
    return results


def A5_abbreviation_expand(question: str, **kwargs) -> List[Dict]:
    """A5: Expand contractions."""
    q_lower = question.lower()
    variants = []
    for contraction, expansion in CONTRACTIONS.items():
        # Case-insensitive search
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        if pattern.search(q_lower):
            new_q = pattern.sub(expansion, question, count=1)
            # Fix capitalization at start
            if question[0].isupper() and new_q[0].islower():
                new_q = new_q[0].upper() + new_q[1:]
            variants.append({
                "variant_question": new_q,
                "replaced_word": contraction,
                "replacement": expansion,
                "rule_id": "A5",
                "rule_name": "Abbreviation Expand",
            })
    return variants


def A6_abbreviation_contract(question: str, **kwargs) -> List[Dict]:
    """A6: Contract expanded forms into abbreviations."""
    variants = []
    for expansion, contraction in EXPANSIONS.items():
        pattern = re.compile(re.escape(expansion), re.IGNORECASE)
        if pattern.search(question):
            new_q = pattern.sub(contraction, question, count=1)
            if question[0].isupper() and new_q[0].islower():
                new_q = new_q[0].upper() + new_q[1:]
            variants.append({
                "variant_question": new_q,
                "replaced_word": expansion,
                "replacement": contraction,
                "rule_id": "A6",
                "rule_name": "Abbreviation Contract",
            })
    return variants


# ============================================================
# GROUP B-Easy: Prefix Insertion
# ============================================================

def B1_test_prefix(question: str, max_variants: int = 3, **kwargs) -> List[Dict]:
    """B1: Add test-like prefix."""
    selected = TEST_PREFIXES[:max_variants]
    variants = []
    for prefix in selected:
        # Lowercase the first letter of original question after prefix
        q_mod = question[0].lower() + question[1:] if question[0].isupper() else question
        new_q = f"{prefix} {q_mod}"
        variants.append({
            "variant_question": new_q,
            "replaced_word": "",
            "replacement": prefix,
            "rule_id": "B1",
            "rule_name": "Test-like Prefix",
        })
    return variants


def B2_conversation_prefix(question: str, max_variants: int = 3, **kwargs) -> List[Dict]:
    """B2: Add conversation-like prefix."""
    selected = CONVERSATION_PREFIXES[:max_variants]
    variants = []
    for prefix in selected:
        q_mod = question[0].lower() + question[1:] if question[0].isupper() else question
        new_q = f"{prefix} {q_mod}"
        variants.append({
            "variant_question": new_q,
            "replaced_word": "",
            "replacement": prefix,
            "rule_id": "B2",
            "rule_name": "Conversation-like Prefix",
        })
    return variants


def B3_instruction_prefix(question: str, max_variants: int = 3, **kwargs) -> List[Dict]:
    """B3: Add instruction-like prefix."""
    selected = INSTRUCTION_PREFIXES[:max_variants]
    variants = []
    for prefix in selected:
        q_mod = question[0].lower() + question[1:] if question[0].isupper() else question
        new_q = f"{prefix} {q_mod}"
        variants.append({
            "variant_question": new_q,
            "replaced_word": "",
            "replacement": prefix,
            "rule_id": "B3",
            "rule_name": "Instruction Prefix",
        })
    return variants


# ============================================================
# GROUP B-Template: WH-word Replacement
# ============================================================

def _find_wh_word(question: str) -> Optional[Tuple[str, int, int]]:
    """Find the WH-word at the beginning of a question.
    Returns (matched_wh, start, end) or None.
    """
    q_lower = question.lower()
    # Check multi-word WH phrases first
    for wh in ["how many", "how much", "how long", "how often", "how old"]:
        if q_lower.startswith(wh):
            return (wh, 0, len(wh))
    # Then single WH words
    for wh in ["who", "what", "where", "when", "why", "how", "which", "whom"]:
        if q_lower.startswith(wh) and (len(q_lower) == len(wh) or not q_lower[len(wh)].isalpha()):
            return (wh, 0, len(wh))
    return None


def _is_wh_determiner(question: str, wh: str) -> bool:
    """Check if 'what/which' is used as a determiner (e.g., 'what color', 'which book').
    In that case, we should only replace the WH-word, not expand it to 'which thing'.
    """
    if wh not in ("what", "which"):
        return False
    rest = question[len(wh):].strip()
    # If followed by a content word (not a verb/aux), it's a determiner
    first_word = rest.split()[0].lower().rstrip("?.,!;:") if rest else ""
    aux_verbs = {"is", "are", "was", "were", "do", "does", "did", "has", "have",
                 "had", "will", "would", "could", "should", "can", "may", "might"}
    # If followed by a noun-like word, it's a determiner
    if first_word and first_word not in aux_verbs and first_word != "about":
        return True
    return False


def B4_wh_expansion(question: str, **kwargs) -> List[Dict]:
    """B4: Expand WH-word to longer phrase."""
    match = _find_wh_word(question)
    if match is None:
        return []
    wh, start, end = match
    if wh not in WH_EXPANSION:
        return []

    # Skip if 'what/which' is used as determiner (e.g., "what color")
    if _is_wh_determiner(question, wh):
        return []

    expansion = WH_EXPANSION[wh]
    # Preserve capitalization
    if question[0].isupper():
        expansion = expansion[0].upper() + expansion[1:]

    new_q = expansion + question[end:]
    return [{
        "variant_question": new_q,
        "replaced_word": question[start:end],
        "replacement": expansion,
        "rule_id": "B4",
        "rule_name": "WH Expansion",
    }]


def B5_wh_rephrase(question: str, max_variants: int = 2, **kwargs) -> List[Dict]:
    """B5: Rephrase WH-word with alternative."""
    match = _find_wh_word(question)
    if match is None:
        return []
    wh, start, end = match
    if wh not in WH_REPHRASE:
        return []

    # For determiner usage, only use "which" <-> "what" swap
    if _is_wh_determiner(question, wh):
        if wh == "what":
            rephrase = "Which"  if question[0].isupper() else "which"
            new_q = rephrase + question[end:]
            return [{
                "variant_question": new_q,
                "replaced_word": question[start:end],
                "replacement": rephrase,
                "rule_id": "B5",
                "rule_name": "WH Rephrase",
            }]
        return []

    variants = []
    rephrases = WH_REPHRASE[wh][:max_variants]
    for rephrase in rephrases:
        if question[0].isupper():
            rephrase = rephrase[0].upper() + rephrase[1:]
        new_q = rephrase + question[end:]
        variants.append({
            "variant_question": new_q,
            "replaced_word": question[start:end],
            "replacement": rephrase,
            "rule_id": "B5",
            "rule_name": "WH Rephrase",
        })
    return variants


# ============================================================
# Registry: all rules
# ============================================================

ALL_RULES = {
    # Group A
    "A1": A1_verb_synonym,
    "A2": A2_noun_synonym,
    "A3": A3_adj_synonym,
    "A4": A4_adv_synonym,
    "A5": A5_abbreviation_expand,
    "A6": A6_abbreviation_contract,
    # Group B-Easy
    "B1": B1_test_prefix,
    "B2": B2_conversation_prefix,
    "B3": B3_instruction_prefix,
    # Group B-Template
    "B4": B4_wh_expansion,
    "B5": B5_wh_rephrase,
}

GROUP_A_RULES = ["A1", "A2", "A3", "A4", "A5", "A6"]
GROUP_B_RULES = ["B1", "B2", "B3", "B4", "B5"]


def generate_all_variants(question: str,
                          rules: Optional[List[str]] = None,
                          max_variants_per_rule: int = 3) -> List[Dict]:
    """
    Apply all (or selected) rules to a question and return all variants.
    Each variant dict contains:
      - rule_id, rule_name
      - variant_question
      - replaced_word, replacement
    """
    if rules is None:
        rules = list(ALL_RULES.keys())

    all_variants = []
    for rule_id in rules:
        if rule_id not in ALL_RULES:
            print(f"Warning: unknown rule '{rule_id}', skipping.")
            continue
        fn = ALL_RULES[rule_id]
        try:
            variants = fn(question, max_variants=max_variants_per_rule)
        except TypeError:
            # Some rules don't take max_variants
            variants = fn(question)
        all_variants.extend(variants)

    return all_variants


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    test_questions = [
        "How does the bulb get its energy supply?",
        "Who beat him up?",
        "what color was it?",
        "What was the name?",
        "Who helped?",
        "In what year?",
        "What's he doing?",
        "Who was it?",
        "What other state ruled it?",
    ]

    print(f"Using WordNet: {USE_WORDNET}")
    print(f"Using spaCy: {USE_SPACY}")
    print("=" * 70)

    for q in test_questions:
        print(f"\nOriginal: {q}")
        variants = generate_all_variants(q)
        if not variants:
            print("  (no variants generated)")
        for v in variants:
            print(f"  [{v['rule_id']}] {v['rule_name']:25s} | {v['variant_question']}")