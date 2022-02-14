from typing import Any, Dict
import difflib
import glob
import itertools
import json
import pickle
import re

import nltk
from nltk.stem import porter
from sklearn.feature_extraction import text as text_extraction

import constants

REMINDER = re.compile(r'\(.*\)')


def GetMaxOracle():
  potential_oracles = glob.glob('oracle-cards-*.json')
  return Oracle(max(potential_oracles))


def MakeCardnamePattern(card: 'Card'):
  if not ' // ' in card.name:
    return MakeFacenamePattern(
        card.name, is_legendary='Legendary' in card.get('type_line', ''))
  return '|'.join(
      MakeFacenamePattern(
          face['name'], is_legendary='Legendary' in face.get('type_line', ''))
      for face in card['card_faces'])


def MakeFacenamePattern(face_name, is_legendary=False):
  pattern = fr'(?:\b{re.escape(face_name)}\b)'
  if is_legendary:
    short_name = fr'(?:\b{re.escape(face_name.split(", ")[0])}\b)'
    pattern += '|' + short_name
  return pattern


class UnknownCardError(Exception):
  """Failed to find card requested for lookup."""


class Card:

  def __init__(self, json_dict):
    self.json = json_dict
    if 'card_faces' in self:
      self['oracle_text'] = '\n'.join(
          face['oracle_text'] for face in self['card_faces'])
      if 'colors' not in self:
        self['colors'] = [
            c for c in constants.WUBRG
            if any(c in face['colors'] for face in self['card_faces'])
        ]
      if 'mana_cost' not in self:
        self['mana_cost'] = self['card_faces'][0]['mana_cost']
    if 'oracle_text' not in self:
      self['oracle_text'] = ''
    cardname_pattern = MakeCardnamePattern(self)
    self['oracle_text'] = re.sub(cardname_pattern, 'CARDNAME',
                                 self['oracle_text'])
    self['oracle_text'] = REMINDER.sub('', self['oracle_text'])

  def __contains__(self, key):
    return key in self.json

  def __getitem__(self, key):
    return self.json[key]

  def get(self, key, default=None):
    return self.json.get(key, default)

  def __setitem__(self, key, value):
    self.json[key] = value

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, o):
    if not isinstance(o, Card):
      return False
    return o.name == self.name

  @property
  def name(self):
    return self.json['name']

  @property
  def shortname(self):
    if len(self.name) <= 16:
      return self.name
    return self.json['name'].split(' // ')[0]


class Oracle:

  def __init__(self, filename):
    """Read all cards from {filename}."""
    self.tfidf_sq = None
    try:
      self.oracle, self.partials = pickle.load(open(f'{filename}.pkl', 'rb'))
    except (IOError, EOFError):
      pass

    json_card_dicts = json.load(open(filename))
    self.oracle = {
        card_dict['name']: Card(card_dict)
        for card_dict in json_card_dicts
        if card_dict['set_type'] not in ('token', 'vanguard', 'memorabilia')
    }

    self.partials = {}
    counter = itertools.count()
    for card in self.oracle.values():
      for part in card.name.split(' // '):
        self.partials[part] = card
      card['index'] = next(counter)
    self.oracle['Life // Death']['mana_cost'] = '{1}{B}'
    assert len(self.oracle) == next(counter)
    pickle.dump((self.oracle, self.partials), open(f'{filename}.pkl', 'wb'))

  def _AllCardAndPartialNames(self):
    all_names = set(self.oracle.keys()) | set(self.partials.keys())
    yield from all_names

  def Get(self, name) -> Card:
    p = self.partials.get(name)
    if p:
      return p
    return self.oracle.get(name)

  def Canonicalize(self, name):
    if name in self.partials:
      return self.partials[name].name
    return name

  def GetClose(self, close_name):
    try:
      return self.Get(close_name)
    except KeyError:
      pass
    try:
      name = difflib.get_close_matches(
          close_name, self._AllCardAndPartialNames(), n=1)[0]
    except IndexError:
      raise UnknownCardError(f'No card found for {close_name:r}.')
    return self.Get(name)

  def GetTfidfSq(self):
    if self.tfidf_sq is not None:
      return self.tfidf_sq
    docs = [
        '\n'.join((
            card['type_line'],
            card['oracle_text'],
        )) for card in self.oracle.values()
    ]

    def stem_tokens(tokens, stemmer):
      stemmed = []
      for item in tokens:
        stemmed.append(stemmer.stem(item))
      return stemmed

    stemmer = porter.PorterStemmer()

    def tokenize(text):
      tokens = nltk.word_tokenize(text)
      stems = stem_tokens(tokens, stemmer)
      return stems

    vectorizer = text_extraction.TfidfVectorizer(
        token_pattern=r'[^\s,.:;—•"]+',
        tokenizer=tokenize,
        stop_words=[
            'a',
            'an',
            'and',
            'of',
            'or',
            'that',
            'the',
            'to',
        ],
        ngram_range=(1, 3),
    )
    tfidf = vectorizer.fit_transform(docs)
    self.tfidf_sq = tfidf.dot(tfidf.T)
    return self.tfidf_sq


def ExpandList(lst):
  """Expand a list by repeating lines that start with a number.

  Example:
      4 Ajani's Pridemate
    becomes
      Ajani's Pridemate
      Ajani's Pridemate
      Ajani's Pridemate
      Ajani's Pridemate

  Args:
    lst: The list to expand.

  Yields:
    The expanded elements.
  """
  for line in lst:
    line = line.strip()
    # line = line.split(' // ')[0]
    try:
      first_token, rest = line.split(maxsplit=1)
    except ValueError:
      yield line
      continue
    if first_token.isnumeric():
      yield from [rest] * int(first_token)
    else:
      yield line
