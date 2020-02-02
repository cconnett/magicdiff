# python3
"""Diffing algorithm for Magic: the Gathering lists."""

import collections
import difflib
import itertools
import json
import math
import pdb
import re
import sys
import traceback
from typing import List, Iterable

from sklearn.feature_extraction import text
import numpy as np
import scipy.optimize

WUBRG = ['W', 'U', 'B', 'R', 'G']


def ColorDistanceOverlap(a: Iterable[str], b: Iterable[str]) -> float:
  a, b = set(a), set(b)
  if not a and a == b:
    return 0
  return 1 - (2 * len(a & b) / (len(a) + len(b)))


def ColorDistanceVector(a: Iterable[str], b: Iterable[str]) -> float:
  a, b = collections.Counter(a), collections.Counter(b)
  if not a or not b:
    if a == b:
      return 0
    else:
      return 1
  dot_product = sum(a[c] * b[c] for c in WUBRG)
  mag_a, mag_b = math.sqrt(sum(a[c]**2 for c in WUBRG)), math.sqrt(
      sum(b[c]**2 for c in WUBRG))
  cosine_dist = dot_product / (mag_a * mag_b)
  return 1 - cosine_dist


ColorDistance = ColorDistanceVector

pip = re.compile(r'\{(.*?)\}')
hybrid = re.compile('([2WUBRG])/([WUBRGP])')

memo = {}


def ManaCostToColorVector(mana_cost: str):
  if mana_cost in memo:
    return memo[mana_cost]
  accumulator = collections.Counter()
  pips = pip.findall(mana_cost)
  for p in pips:
    if p in WUBRG:
      accumulator[p] += 1
    elif hybrid.match(p):
      left, right = hybrid.match(p).groups()
      if right == 'P':
        accumulator[left] += 0.33
      elif left == '2':
        accumulator[right] += 0.67
      else:
        accumulator[left] += 0.5
        accumulator[right] += 0.5
    elif p == 'X':
      accumulator['C'] += 3
    else:
      accumulator['C'] += int(p)
  vector = np.array(
      [
          accumulator['W'],
          accumulator['U'],
          accumulator['B'],
          accumulator['R'],
          accumulator['G'],
          # Colorless is not a color ;-)
      ],
      dtype=float)
  if not vector.any():
    vector = np.array([1, 1, 1, 1, 1], dtype=float)
  vector /= np.linalg.norm(vector)
  vector *= sum(accumulator.values())
  memo[mana_cost] = vector
  return vector


def TextDistanceGestalt(a: str, b: str) -> float:
  return 1 - difflib.SequenceMatcher(a=a, b=b).ratio()


def TextDistanceTfidf(a: str, b: str) -> float:
  pass


def TypeBucket(types: List[str]) -> str:
  if 'Land' in types:
    return 'Land'
  elif 'Creature' in types:
    return 'Creature'
  elif 'Instant' in types or 'Sorcery' in types:
    return 'Spell'
  else:
    return 'Permanent'


def TypesDistance(a: List[str], b: List[str]) -> int:
  return 1 - bool(TypeBucket(a) == TypeBucket(b))


def CmcMetric(card):
  return card['convertedManaCost'] + (4 if '{X}' in card.get('manaCost', '')
                                      else 0)


def GirthInt(value: str) -> int:
  try:
    return int(value)
  except ValueError:
    if '*' in value:
      return 4
    return 0


def GirthDistance(a, b):
  a_girth = (GirthInt(a.get('power', a['convertedManaCost'])) +
             GirthInt(a.get('toughness', a['convertedManaCost'])))
  b_girth = (GirthInt(b.get('power', b['convertedManaCost'])) +
             GirthInt(b.get('toughness', b['convertedManaCost'])))
  return 1 - math.exp(-abs(a_girth - b_girth) / 3)


def CardDistance(tfidf, a, b):
  color = ColorDistance(a['colors'], b['colors'])
  color_identity = ColorDistance(a['colorIdentity'], b['colorIdentity'])
  mana_cost = 1 - math.exp(-np.linalg.norm(
      ManaCostToColorVector(a.get('manaCost', '')) -
      ManaCostToColorVector(b.get('manaCost', ''))) / 3)
  # text = TextDistanceGestalt(a['text'], b['text'])
  text_product = tfidf[a['index']].dot(tfidf[b['index']].T)
  if text_product:
    text = 1 - text_product.data[0]
  else:
    text = 1
  types = TypesDistance(a['types'], b['types'])
  girth = GirthDistance(a, b)

  weights = np.array([1, 2, 3, 1.4, 0.6, 0.5])
  metrics = np.array([color, color_identity, mana_cost, text, types, girth])

  return weights.dot(metrics.T**2)


reminder = re.compile(r'\(.*\)')


def GetCards():
  """Read all cards from AllCards.json."""
  c = json.load(open('AllCards.json'))
  cards = {}
  counter = itertools.count()
  for card in c.values():
    if (card['layout'] in ('split', 'aftermath') and
        'Adventure' not in card['subtypes']):
      name = ' // '.join(card['names'])
    else:
      name = card['name']
    card['text'] = re.sub(r'\b' + re.escape(name) + r'\b', 'CARDNAME',
                          card.get('text', ''))
    # card['text'] = reminder.sub('', card['text'])
    if 'names' in card:
      text = ''
      for nm in card['names']:
        text += '\n' + c[nm].get('text', '')
        card['colors'] = list(set(card['colors']) | set(c[nm]['colors']))
      card['text'] = text
    if name not in cards:
      cards[name] = card
      card['index'] = next(counter)
  assert len(cards) == next(counter)
  assert len(cards) == len(set(cards))
  return cards


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


def CubeDiff(card_data, tfidf, list_a, list_b):
  """Yield a diff between lists by linear sum assignment."""
  set_a = collections.Counter(ExpandList(list_a))
  set_b = collections.Counter(ExpandList(list_b))
  removes = list((set_a - set_b).elements())
  adds = list((set_b - set_a).elements())

  n, m = len(removes), len(adds)
  costs = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
      costs[i, j] = CardDistance(tfidf, card_data[removes[i]],
                                 card_data[adds[j]])
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


def FormatDiff(diff):
  width_removes = max((len(r) for r, a in diff if r), default=0)
  width_adds = max((len(a) for r, a in diff if a), default=0)
  for remove, add in diff:
    if remove and add:
      yield f'{remove:{width_removes}} -> {add:{width_adds}}'
    elif remove:
      yield f'- {remove}'
    else:
      yield f'{"":{width_removes}}  + {add}'


def main(argv):
  card_data = GetCards()
  docs = [
      '\n'.join([
          ' '.join(f'istype@{t}' for t in card['types']),
          ' '.join(f'subtype@{t}' for t in card['subtypes']), card['text']
      ]) for card in card_data.values()
  ]
  tfidf = text.TfidfVectorizer().fit_transform(docs)

  list_a = [line.strip() for line in open(argv[1]).readlines()]
  list_b = [line.strip() for line in open(argv[2]).readlines()]

  def SortKey(change):
    card_a, card_b = change
    if not card_a:
      card_a = card_b
    colors = card_data[card_a]['colors']
    ci = card_data[card_a]['colorIdentity']
    return (
        len(colors),
        sorted(tuple(WUBRG.index(c) for c in colors)),
        sorted(tuple(WUBRG.index(c) for c in ci)),
        int(card_data[card_a]['convertedManaCost']),
        card_a,
    )

  diff = list(CubeDiff(card_data, tfidf, list_a, list_b))
  diff = sorted(diff, key=SortKey)
  for line in FormatDiff(diff):
    print(line)


if __name__ == '__main__':
  try:
    main(sys.argv)
  except Exception as e:  # pylint: disable=broad-except
    traceback.print_tb(e.__traceback__)
    print(repr(e))
    pdb.post_mortem()
