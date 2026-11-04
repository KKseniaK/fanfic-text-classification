"""The original deterministic NLTK preprocessing, with frozen notebook resources."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.model_selection import StratifiedGroupKFold

RESOURCE = Path(__file__).parent / 'resources/preprocessing.json'
RULES = json.loads(RESOURCE.read_text(encoding='utf-8'))
STOP_WORDS = set(RULES['stop_words'])
SPECIAL_FORMS = RULES['special_forms']
STEMMER = RussianStemmer()

@lru_cache(maxsize=100_000)
def stem(word):
    return SPECIAL_FORMS[word] if word in SPECIAL_FORMS else STEMMER.stem(word)

def preprocess(text):
    words = [t.lower() for t in wordpunct_tokenize(text)]
    return ' '.join(stem(w) for w in words if w.isalpha() and w not in STOP_WORDS)

def boolean(series):
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    if not normalized.isin(mapping).all():
        raise ValueError(f'Invalid boolean values in {series.name}')
    return normalized.map(mapping).astype(bool)

def load_task(path):
    frame = pd.read_csv(path, dtype={'story_id':str, 'author_url':str})
    required = {'story_id','author_url','text','hp_eligible','translation_badge_present','label_angst_tag','label_humor_tag'}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    hp = frame.loc[boolean(frame.hp_eligible) & frame.text.notna()].copy()
    angst = pd.to_numeric(hp.label_angst_tag, errors='raise')
    humor = pd.to_numeric(hp.label_humor_tag, errors='raise')
    if not angst.isin([0,1]).all() or not humor.isin([0,1]).all():
        raise ValueError('Genre labels must be 0 or 1')
    task = hp.loc[~boolean(hp.translation_badge_present) & ((angst + humor) == 1)].copy()
    task['target'] = task.label_humor_tag.astype(int)
    task['text_stem'] = task.text.map(preprocess)
    task = task.loc[task.text_stem.ne('')].reset_index(drop=True)
    if task.author_url.isna().any() or task.author_url.str.strip().eq('').any():
        raise ValueError('Author groups are missing; cannot safely reproduce grouped split')
    if task.story_id.isna().any() or task.story_id.duplicated().any():
        raise ValueError('Missing or duplicated story_id')
    return task, {'corpus_rows':len(frame), 'hp_with_text':len(hp), 'task_rows':len(task)}

def outer_split(task):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(splitter.split(task.text_stem, task.target, groups=task.author_url))
    train, test = task.iloc[tr].copy(), task.iloc[te].copy()
    check_split(train, test)
    return train, test

def check_split(left, right):
    for column in ['author_url','story_id','text_stem']:
        if set(left[column]) & set(right[column]):
            raise ValueError(f'Overlap across partitions: {column}. Group duplicates before evaluation.')
    if set(left.target) != {0,1} or set(right.target) != {0,1}:
        raise ValueError('Both classes are required in every partition')

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
