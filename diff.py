# python3
"""Diffing algorithm for Magic: the Gathering lists."""

from typing import List, Iterable
import collections
import glob
import itertools
import json
import math
import pdb
import pickle
import re
import sys
import traceback

import numpy as np
import scipy.optimize
from sklearn.feature_extraction import text as text_extraction

import constants
import color_distance
import mana_cost_distance

REMINDER = re.compile(r'\(.*\)')


def GetCards(filename):
  """Read all cards from AllCards.json."""
  try:
    return pickle.load(open(f'{filename}.pkl', 'rb'))
  except (IOError, EOFError):
    pass

  card_list = json.load(open(filename))
  card_map = {
      card['name']: card
      for card in card_list
      if card['set_type'] not in ('token', 'vanguard', 'memorabilia')
  }
  partial_names = {}
  counter = itertools.count()
  for card in card_map.values():
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
        partial_names[part] = card
    else:
      cardname_pattern = card['name']
    card['oracle_text'] = re.sub(fr'\b{re.escape(cardname_pattern)}\b',
                                 'CARDNAME', card['oracle_text'])
    card['oracle_text'] = REMINDER.sub('', card['oracle_text'])
    card['index'] = next(counter)
  card_map['Life // Death']['mana_cost'] = '{1}{B}'
  assert len(card_map) == next(counter)
  pickle.dump((card_map, partial_names), open(f'{filename}.pkl', 'wb'))
  return card_map, partial_names


ORACLE, PARTIALS = None, None


def TypeBucket(types: List[str]) -> str:
  if 'Land' in types:
    return 'Land'
  elif 'Creature' in types:
    return 'Creature'
  elif 'Instant' in types or 'Sorcery' in types:
    return 'Spell'
  elif 'Planeswalker' in types:
    return 'Planeswalker'
  else:
    return 'Permanent'


def TypesDistance(a: List[str], b: List[str]) -> int:
  return 1 - bool(TypeBucket(a) == TypeBucket(b))


def Metrics(tfidf_sq, a, b):
  """A metric for difference between cards a and b."""
  color = color_distance.EditDistance(a['colors'], b['colors'])
  color_identity = color_distance.EditDistance(a['color_identity'],
                                               b['color_identity'])
  mana_cost = mana_cost_distance.EditDistance(
      a.get('mana_cost', ''), b.get('mana_cost', ''))
  text_product = tfidf_sq[a['index'], b['index']]
  text = 1 - text_product
  types = TypesDistance(a['type_line'], b['type_line'])

  metrics = np.array([color, color_identity, mana_cost, text, types])
  return metrics


def CardDistance(tfidf_sq, a, b):
  """A metric for difference between cards a and b."""
  color, color_identity, mana_cost, text, types = Metrics(tfidf_sq, a, b)

  weights = np.array([1, 2, 3, 1.4, 0.6])
  metrics = Metrics(tfidf_sq, a, b)
  return weights.dot(metrics.T**2)


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


def GetCosts(tfidf_sq, set_a, set_b):
  n, m = len(set_a), len(set_b)
  costs = np.zeros((n, m))
  for i in range(n):
    for j in range(i, m):
      a = ORACLE.get(set_a[i], PARTIALS.get(set_a[i]))
      b = ORACLE.get(set_b[j], PARTIALS.get(set_b[j]))
      costs[i, j] = CardDistance(tfidf_sq, a, b)
      try:
        costs[j, i] = costs[i, j]
      except IndexError:
        pass
  return costs


def FormatCosts(costs):
  n, m = costs.shape
  for i in range(n):
    print(' ' * 6 * i, end='')
    for j in range(i + 1, m):
      c = int(costs[i, j] * 10000)
      print(f'{c:5d}', end=' ')
    print()


def CubeDiff(tfidf_sq, list_a, list_b):
  """Yield a diff between lists by linear sum assignment."""
  set_a = collections.Counter(ExpandList(list_a))
  set_b = collections.Counter(ExpandList(list_b))
  removes = list((set_a - set_b).elements())
  adds = list((set_b - set_a).elements())
  n, m = len(removes), len(adds)

  costs = GetCosts(tfidf_sq, removes, adds)
  rows, cols = scipy.optimize.linear_sum_assignment(costs)
  diff = zip(rows, cols)
  for remove, add in diff:
    yield (removes[remove], adds[add])
  if n > m:
    for extra_remove in set(range(n)) - set(rows):
      yield (removes[extra_remove], None)
  if n < m:
    for extra_add in set(range(m)) - set(cols):
      yield (None, adds[extra_add])


def TextDiff(diff):
  width_removes = max((len(r) for r, a in diff if r), default=0)
  width_adds = max((len(a) for r, a in diff if a), default=0)
  for remove, add in diff:
    if remove and add:
      yield f'  {remove:{width_removes}} -> {add:{width_adds}}'
    elif remove:
      yield f'- {remove}'
    else:
      yield f'{"":{width_removes}}  + {add}'


def CardImg(imagery, name):
  if name == 'REMOVED':
    return '<img class="card" src="BurnCard.png">'
  elif name == 'ADDED':
    return '<img class="card" src="UnburnCard.png">'
  elif name in imagery:
    return f'<img class="card" src="{imagery[name]}">'
  elif name in PARTIALS:
    key = PARTIALS[name]['name']
    return f'<img class="card" src="{imagery[key]}">'
  else:
    return name


def GetImagery():
  """Get the imagery dictionary."""
  imagery = {
      card['name']: card['image_uris']['small']
      for card in ORACLE.values()
      if 'image_uris' in card
  }
  imagery.update({
      card['name']: card['card_faces'][0]['image_uris']['small']
      for card in ORACLE.values()
      if 'card_faces' in card and 'image_uris' not in card
  })
  assert imagery
  return imagery


def PageDiff(diff):
  """Generate an HTML diff."""
  imagery = GetImagery()

  yield '<html>'
  yield f'<head><style>{constants.CSS}</style>'
  yield '<link rel="icon" src="icon.png"></head>'
  yield '<body><ul>'
  for remove, add in diff:
    yield '<li class="change">'
    if remove and add:
      icon = '<img class="change-icon" src="Change.png">'
    elif add:
      icon = '<img class="change-icon" src="Plus.png">'
    elif remove:
      icon = '<img class="change-icon" src="Minus.png">'

    yield CardImg(imagery, remove or 'ADDED')
    yield icon
    yield CardImg(imagery, add or 'REMOVED')
    yield '</li>'
  yield '</ul></body></html>'


def Canonicalize(name):
  if name in PARTIALS:
    return PARTIALS[name]['name']
  return name


def main(argv):
  global ORACLE, PARTIALS
  potential_oracles = glob.glob('oracle-cards-*.json')
  ORACLE, PARTIALS = GetCards(max(potential_oracles))
  docs = [
      '\n'.join((
          card['type_line'],
          card['oracle_text'],
      )) for card in ORACLE.values()
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
  tfidf_sq = tfidf * tfidf.T

  list_a = [
      Canonicalize(line.strip())
      for line in ExpandList(open(argv[1]).readlines())
  ]
  list_b = [
      Canonicalize(line.strip())
      for line in ExpandList(open(argv[2]).readlines())
  ]

  def SortKey(change):
    card_a, card_b = change
    if not card_a:
      card_a = card_b
    card_a = ORACLE.get(card_a, PARTIALS.get(card_a))
    colors = card_a['colors']
    ci = card_a['color_identity']
    return (
        len(colors),
        sorted(tuple(constants.WUBRG.index(c) for c in colors)),
        sorted(tuple(constants.WUBRG.index(c) for c in ci)),
        int(card_a['cmc']),
        card_a['name'],
    )

  # costs = GetCosts(tfidf_sq, list_a, list_a)
  # FormatCosts(costs)
  # return
  diff = list(CubeDiff(tfidf_sq, list_a, list_b))
  diff = sorted(diff, key=SortKey)
  # for line in PageDiff(diff):
  #   print(line)
  for line in TextDiff(diff):
    print(line)


if __name__ == '__main__':
  try:
    main(sys.argv)
  except Exception as e:
    traceback.print_tb(e.__traceback__)
    print(repr(e))
    pdb.post_mortem()
    raise
