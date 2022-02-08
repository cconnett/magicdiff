# python3
"""Diffing algorithm for Magic: the Gathering lists."""

from typing import List, Iterable
import collections
import functools
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

WUBRG = ['W', 'U', 'B', 'R', 'G']

REMINDER = re.compile(r'\(.*\)')

CSS = '''
li {
    display: flex;
    align-items: center;
    margin: 0 3em 2em 0;
}

ul {
    list-style-type: none;
    display: flex;
    flex-flow: row wrap;
    justify-content: space-between;
}

img.change-icon {
    margin: 10;
}
img.card {
    width: 146;
}
'''


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
            c for c in WUBRG
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


def ColorDistanceOverlap(a: Iterable[str], b: Iterable[str]) -> float:
  a, b = set(a), set(b)
  if not a and a == b:
    return 0
  return 1 - (2 * len(a & b) / (len(a) + len(b)))


def ColorDistanceVector(a: Iterable[str], b: Iterable[str]) -> float:
  """Distance between color/identity by vector cosine distance."""
  a, b = collections.Counter(a), collections.Counter(b)
  if not a or not b:
    if a == b:
      return 0
    else:
      return 1
  dot_product = sum(a[c] * b[c] for c in WUBRG)
  mag_a = math.sqrt(sum(a[c]**2 for c in WUBRG))
  mag_b = math.sqrt(sum(b[c]**2 for c in WUBRG))
  cosine_dist = dot_product / (mag_a * mag_b)
  return 1 - cosine_dist


def ColorDistanceEdit(a, b):
  distance = 0
  if ('W' in a) != ('W' in b):
    distance += 1
  if ('U' in a) != ('U' in b):
    distance += 1
  if ('B' in a) != ('B' in b):
    distance += 1
  if ('R' in a) != ('R' in b):
    distance += 1
  if ('G' in a) != ('G' in b):
    distance += 1
  return distance


ColorDistance = ColorDistanceEdit

PIP = re.compile(r'\{(.*?)\}')
HYBRID = re.compile('([2WUBRG])/([WUBRGP])')


@functools.cache
def FlattenManaCost(mana_cost: str):
  """Reduce a mana cost down to array: generic, W pips, U, B, R, G."""
  mana_cost = ''.join(mana_cost.split(' // '))
  accumulator = collections.Counter()
  pips = PIP.findall(mana_cost)
  for p in pips:
    if p in WUBRG:
      accumulator[p] += 1
    elif HYBRID.match(p):
      left, right = HYBRID.match(p).groups()
      if right == 'P':
        accumulator[left] += 0.33
      elif left == '2':
        accumulator[right] += 0.67
      else:
        accumulator[left] += 0.5
        accumulator[right] += 0.5
    elif p in 'XYZ':
      accumulator['V'] += 3
    elif p in ('C', 'S'):  # Colorless cost; snow
      accumulator['V'] += 1  # Mana value accumulator
    elif p.startswith('H'):  # Half mana
      accumulator[p[1]] += 0.5
    else:
      accumulator['V'] += int(p)
  vector = np.array([
      accumulator['W'],
      accumulator['U'],
      accumulator['B'],
      accumulator['R'],
      accumulator['G'],
      accumulator['V'],
  ],
                    dtype=float)
  return vector


def ManaCostToColorVector(mana_cost: str):
  """Convert a mana cost to a vector in colorspace."""
  vector = FlattenManaCost(mana_cost)
  if not vector.any():
    vector = np.array([0, 0, 0, 0, 0, 1], dtype=float)
  vector /= np.linalg.norm(vector)
  vector *= vector[5]
  return vector


def ManaCostEditDistance(mana_cost_a: str, mana_cost_b: str):
  """Distance between two mana costs by edit distance."""
  ret = sum(abs(FlattenManaCost(mana_cost_a) - FlattenManaCost(mana_cost_b)))
  # print(mana_cost_a, mana_cost_b, ret)
  return ret


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


def Metrics(tfidf_sq, a, b):
  """A metric for difference between cards a and b."""
  color = ColorDistance(a['colors'], b['colors'])
  color_identity = ColorDistance(a['color_identity'], b['color_identity'])
  mana_cost = 1 - math.exp(-np.linalg.norm(
      ManaCostToColorVector(a.get('mana_cost', '')) -
      ManaCostToColorVector(b.get('mana_cost', ''))) / 3)
  text_product = tfidf_sq[a['index'], b['index']]
  text = 1 - text_product
  types = TypesDistance(a['type_line'], b['type_line'])
  girth = GirthDistance(a, b)

  metrics = np.array([color, color_identity, mana_cost, text, types, girth])
  return metrics


def CardDistance(tfidf_sq, a, b):
  """A metric for difference between cards a and b."""
  color, color_identity, mana_cost, text, types, girth = Metrics(tfidf_sq, a, b)

  weights = np.array([1, 2, 3, 1.4, 0.6, 0.5])
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
    yield (removes[remove], adds[add],
           Metrics(tfidf_sq,
                   ORACLE.get(removes[remove], PARTIALS.get(removes[remove])),
                   ORACLE.get(adds[add],
                              PARTIALS.get(adds[add]))), costs[remove, add])
  if n > m:
    for extra_remove in set(range(n)) - set(rows):
      yield (removes[extra_remove], None)
  if n < m:
    for extra_add in set(range(m)) - set(cols):
      yield (None, adds[extra_add])


def TextDiff(diff):
  width_removes = max((len(r) for r, a, s, _ in diff if r), default=0)
  width_adds = max((len(a) for r, a, s, _ in diff if a), default=0)
  for remove, add, metrics, score in diff:
    if remove and add:
      score_string = ', '.join(f'{m:4.2f}' for m in metrics)
      yield f'{score:5.1f} ({score_string}) {remove:{width_removes}} -> {add:{width_adds}}'
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
  yield f'<head><style>{CSS}</style><link rel="icon" src="icon.png"</head>'
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
    card_a, card_b, score, _ = change
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
