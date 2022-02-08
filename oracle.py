from typing import Any, Dict
import glob
import json
import pickle
import itertools
import re

from sklearn.feature_extraction import text as text_extraction

import constants
# Type for Card struct from Oracle json: Dictionary of strings to stuff.
Card = Dict[str, Any]

REMINDER = re.compile(r'\(.*\)')


def GetMaxOracle():
  potential_oracles = glob.glob('oracle-cards-*.json')
  return Oracle(max(potential_oracles))


class Oracle:

  def __init__(self, filename):
    """Read all cards from {filename}."""
    self.tfidf_sq = None
    try:
      self.oracle, self.partials = pickle.load(open(f'{filename}.pkl', 'rb'))
    except (IOError, EOFError):
      pass

    card_list = json.load(open(filename))
    self.oracle = {
        card['name']: card
        for card in card_list
        if card['set_type'] not in ('token', 'vanguard', 'memorabilia')
    }
    self.partials = {}
    counter = itertools.count()
    for card in self.oracle.values():
      if 'card_faces' in card:
        card['oracle_text'] = '\n'.join(
            face['oracle_text'] for face in card['card_faces'])
        if 'colors' not in card:
          card['colors'] = [
              c for c in constants.WUBRG
              if any(c in face['colors'] for face in card['card_faces'])
          ]
        if 'mana_cost' not in card:
          card['mana_cost'] = card['card_faces'][0]['mana_cost']
      if 'oracle_text' not in card:
        card['oracle_text'] = ''
      if ' // ' in card['name']:
        cardname_pattern = '|'.join(
            re.escape(part) for part in card['name'].split(' // '))
        for part in card['name'].split(' // '):
          self.partials[part] = card
      else:
        cardname_pattern = card['name']
      card['oracle_text'] = re.sub(fr'\b{re.escape(cardname_pattern)}\b',
                                   'CARDNAME', card['oracle_text'])
      card['oracle_text'] = REMINDER.sub('', card['oracle_text'])
      card['index'] = next(counter)
    self.oracle['Life // Death']['mana_cost'] = '{1}{B}'
    assert len(self.oracle) == next(counter)
    pickle.dump((self.oracle, self.partials), open(f'{filename}.pkl', 'wb'))

  def Get(self, name) -> Card:
    p = self.partials.get(name)
    if p:
      return p
    return self.oracle.get(name)

  def Canonicalize(self, name):
    if name in self.partials:
      return self.partials[name]['name']
    return name

  def GetTfidfSq(self):
    if self.tfidf_sq is not None:
      return self.tfidf_sq
    docs = [
        '\n'.join((
            card['type_line'],
            card['oracle_text'],
        )) for card in self.oracle.values()
    ]
    vectorizer = text_extraction.TfidfVectorizer(
        token_pattern=r'[^\s,.:;—•"]+',
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
        ngram_range=(2, 3),
    )
    tfidf = vectorizer.fit_transform(docs)
    self.tfidf_sq = tfidf * tfidf.T
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
