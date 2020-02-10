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

import numpy as np
import scipy.optimize
from sklearn.feature_extraction import text as text_extraction

WUBRG = ['W', 'U', 'B', 'R', 'G']

REMINDER = re.compile(r'\(.*\)')


def GetCards():
  """Read all cards from AllCards.json."""
  card_list = json.load(open('scryfall-oracle-cards.json'))
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
            c for c in WUBRG
            if any(c in face['colors'] for face in card['card_faces'])
        ]
    if 'oracle_text' not in card:
      card['oracle_text'] = ''
    if ' // ' in card['name']:
      cardname_pattern = '|'.join(
          re.escape(part) for part in card['name'].split(' // '))
      for part in card['name'].split(' // '):
        partial_names[part] = card
    else:
      cardname_pattern = card['name']
    card['oracle_text'] = re.sub(r'\b' + cardname_pattern + r'\b', 'CARDNAME',
                                 card['oracle_text'])
    card['oracle_text'] = REMINDER.sub('', card['oracle_text'])
    card['index'] = next(counter)
  assert len(card_map) == next(counter)
  return card_map, partial_names


ORACLE, PARTIALS = None, None


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
  return card['cmc'] + (4 if '{X}' in card.get('mana_cost', '') else 0)


def GirthInt(value: str) -> int:
  try:
    return int(value)
  except ValueError:
    if '*' in value:
      return 4
    return 0


def GirthDistance(a, b):
  a_girth = (
      GirthInt(a.get('power', a['cmc'])) +
      GirthInt(a.get('toughness', a['cmc'])))
  b_girth = (
      GirthInt(b.get('power', b['cmc'])) +
      GirthInt(b.get('toughness', b['cmc'])))
  return 1 - math.exp(-abs(a_girth - b_girth) / 3)


def CardDistance(tfidf_sq, a, b):
  color = ColorDistance(a['colors'], b['colors'])
  color_identity = ColorDistance(a['color_identity'], b['color_identity'])
  mana_cost = 1 - math.exp(-np.linalg.norm(
      ManaCostToColorVector(a.get('mana_cost', '')) -
      ManaCostToColorVector(b.get('mana_cost', ''))) / 3)
  # text = TextDistanceGestalt(a['text'], b['text'])
  text_product = tfidf_sq[a['index'], b['index']]
  text = 1 - text_product
  types = TypesDistance(a['type_line'], b['type_line'])
  girth = GirthDistance(a, b)

  weights = np.array([1, 2, 3, 1.4, 0.6, 0.5])
  metrics = np.array([color, color_identity, mana_cost, text, types, girth])

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


def CubeDiff(tfidf_sq, list_a, list_b):
  """Yield a diff between lists by linear sum assignment."""
  set_a = collections.Counter(ExpandList(list_a))
  set_b = collections.Counter(ExpandList(list_b))
  removes = list((set_a - set_b).elements())
  adds = list((set_b - set_a).elements())

  n, m = len(removes), len(adds)
  costs = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
      remove = ORACLE.get(removes[i], PARTIALS.get(removes[i]))
      add = ORACLE.get(adds[j], PARTIALS.get(adds[j]))
      costs[i, j] = CardDistance(tfidf_sq, remove, add)
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


def PageDiff(diff):
  """Generate an HTML diff."""
  imagery = {
      card['name']: card['image_uris']['small']
      for card in ORACLE.values()
      if 'image_uris' in card
  }
  assert imagery
  yield '<html><body><table><tr><th>Removed</th><th>Added</th></tr>'
  for remove, add in diff:
    yield '<tr><td>'
    if remove:
      if remove not in imagery:
        yield remove
      else:
        yield f'<img src="{imagery[remove]}">'
    else:
      yield 'Added'
    yield '</td><td>'
    if add:
      if add not in imagery:
        yield add
      else:
        yield f'<img src="{imagery[add]}">'
    else:
      yield 'Removed'
    yield '</td></tr>'
  yield '</table></body></html>'


def main(argv):
  global ORACLE, PARTIALS
  ORACLE, PARTIALS = GetCards()
  docs = [
      '\n'.join((
          card['type_line'],
          card['oracle_text'],
      )) for card in ORACLE.values()
  ]
  # tfidf = text_extraction.TfidfVectorizer().fit_transform(docs)
  tfidf = text_extraction.TfidfVectorizer(
      token_pattern=r'[^\s,.:;—•]+', ngram_range=(2, 3),
      max_features=3000).fit_transform(docs)
  tfidf_sq = tfidf * tfidf.T

  list_a = [line.strip() for line in open(argv[1]).readlines()]
  list_b = [line.strip() for line in open(argv[2]).readlines()]

  def SortKey(change):
    card_a, card_b = change
    if not card_a:
      card_a = card_b
    card_a = ORACLE.get(card_a, PARTIALS.get(card_a))
    colors = card_a['colors']
    ci = card_a['color_identity']
    return (
        len(colors),
        sorted(tuple(WUBRG.index(c) for c in colors)),
        sorted(tuple(WUBRG.index(c) for c in ci)),
        int(card_a['cmc']),
        card_a['name'],
    )

  diff = list(CubeDiff(tfidf_sq, list_a, list_b))
  diff = sorted(diff, key=SortKey)
  for line in PageDiff(diff):
    print(line)


if __name__ == '__main__':
  try:
    main(sys.argv)
  except Exception as e:  # pylint: disable=broad-except
    traceback.print_tb(e.__traceback__)
    print(repr(e))
    pdb.post_mortem()
