from typing import Any, Dict
import difflib
import glob
import itertools
import json
import pickle
import re
import sys

import h5py

import constants

REMINDER = re.compile(r'\(.*\)')
TFIDF_FILENAME = '/tmp/tfidf.hdf5'


def GetMaxOracle():
  potential_oracles = glob.glob('oracle-cards-*.json')
  return Oracle(max(potential_oracles))


def GetLiteOracle():
  return pickle.load(open('lite-oracle.pkl', 'rb'))


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

  def __init__(self, name, json_string):
    self.name = name
    self.index = None
    self.json_string = json_string
    self.json = {}

  def Parse(self):
    if self.json:
      return
    self._RealParse()

  def _RealParse(self):
    self.json = json.loads(self.json_string)
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

  def __str__(self):
    return f'Card({self.shortname}, {self.index})'

  @property
  def shortname(self):
    if len(self.name) <= 16:
      return self.name
    return self.name.split(' // ')[0]


NAME_PATTERN = re.compile('"name":"(.*?)",')
SET_TYPE_PATTERN = re.compile('"set_type":"(.*?)",')


class Oracle:

  def __init__(self, filename):
    """Read all cards from {filename}."""
    self.tfidf_sq = None
    self.oracle = {}
    self.partials = {}
    self.all_names = set()

    lines = open(filename).readlines()
    lines = lines[1:-1]  # Remove opening and closing square brackets.
    for line in lines:
      match = SET_TYPE_PATTERN.search(line)
      assert match
      if match.group(1) in ('token', 'vanguard', 'memorabilia'):
        continue
      match = NAME_PATTERN.search(line)
      assert match
      name = match.group(1)
      line = line.strip(',\n')
      self.oracle[name] = Card(name, line)

    counter = itertools.count()
    for name, card in self.oracle.items():
      card.index = next(counter)
      self.all_names.add(name)
      for part in name.split(' // '):
        self.partials[part] = card
        self.all_names.add(part)

  def _ParseAll(self):
    # This should only be needed for generating the tf-idf matrix.
    for card in self.oracle.values():
      card.Parse()

  def Get(self, name) -> Card:
    card = self.partials.get(name)
    if not card:
      card = self.oracle.get(name)
    return card

  def GetClose(self, close_name):
    name = self.Get(close_name)
    if name is not None:
      return name
    try:
      name = difflib.get_close_matches(close_name, self.all_names, n=1)[0]
      print(f'Corrected {close_name} -> {name}', file=sys.stderr)
    except IndexError:
      raise UnknownCardError(f'No card found for {close_name:r}.') from None
    return self.Get(name)

  def GetTfidfSq(self):
    if self.tfidf_sq is not None:
      return self.tfidf_sq
    try:
      self._LoadTfidfSq()
    except IOError:
      self._WriteTfidfFile()
      self._LoadTfidfSq()
    return self.tfidf_sq

  def _WriteTfidfFile(oracle):
    print('Creating tf-idf matrix.', file=sys.stderr)
    oracle._CreateTfidfSq()
    print('Writing tf-idf matrix.', file=sys.stderr)
    with h5py.File(TFIDF_FILENAME, 'w') as f:
      f.create_dataset('tfidf', data=oracle.tfidf_sq.todense())
    print('Wrote tf-idf matrix.', file=sys.stderr)

  def _LoadTfidfSq(self):
    self.tfidf_sq = h5py.File(TFIDF_FILENAME)['tfidf']

  def _CreateTfidfSq(self):
    import nltk
    from nltk.stem import porter
    from sklearn.feature_extraction import text as text_extraction

    self._ParseAll()
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
    try:
      first_token, rest = line.split(maxsplit=1)
    except ValueError:
      yield line
      continue
    if first_token.isnumeric():
      yield from [rest] * int(first_token)
    else:
      yield line


DECK_TERMS = ('deck', 'maindeck', 'sideboard', 'side', 'main')
SECTION_PATTERN = re.compile(r'(#+)\s*(.+)')


class CardListSection:

  def __init__(self, name, depth):
    self.depth = depth
    self.name = name
    self.cards = []

  def append(self, card):
    self.cards.append(card)


class CardList:

  def __init__(self, lines, oracle):
    section = CardListSection('', 0)
    self.sections = {'': section}
    for line in lines:
      line = line.strip()
      if not line:
        continue
      elif line.lower() in DECK_TERMS:
        section = self.sections.setdefault(line, CardListSection(line, 1))
        continue
      elif match := SECTION_PATTERN.fullmatch(line):
        hashes, name = match.groups()
        section = self.sections.setdefault(name,
                                           CardListSection(name, len(hashes)))
      else:
        multiplicity = 1
        name = line
        try:
          first_token, rest = line.split(maxsplit=1)
          if first_token.isnumeric():
            multiplicity = int(first_token)
            name = rest
        except ValueError:
          pass

        card = oracle.GetClose(name)
        for _ in range(multiplicity):
          section.append(card)
