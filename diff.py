# python3
"""Diffing algorithm for Magic: the Gathering lists."""
import collections
import difflib
import json
import pdb
import re
import sys
import traceback
from typing import List

import numpy
import scipy.optimize

WUBRG = ['W', 'U', 'B', 'R', 'G']


def ColorDistance(a: List[str], b: List[str]) -> float:
  a, b = set(a), set(b)
  if not a and a == b:
    return 0
  return 1 - (2 * len(a & b) / (len(a) + len(b)))


def TextDistance(a: str, b: str) -> float:
  return 1 - difflib.SequenceMatcher(a=a, b=b).ratio()


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


def CardDistanceFeatures(a, b):
  color = ColorDistance(a['colors'], b['colors'])
  color_identity = ColorDistance(a['colorIdentity'], b['colorIdentity'])
  text = TextDistance(a['text'], b['text'])
  types = TypesDistance(a['types'], b['types'])
  cmc = min(4, abs(CmcMetric(a) - CmcMetric(b))) / 4

  return [
      2 * color,
      2 * color_identity,
      2 * text,
      0.3 * types,
      1.5 * cmc,
  ]


def CardDistance(a, b):
  features = CardDistanceFeatures(a, b)
  return sum(features) / len(features)


def GetCards():
  """Read all cards from AllCards.json."""
  c = json.load(open('AllCards.json'))
  cards = {}
  for card in c.values():
    if card['layout'] == 'split':
      name = ' // '.join(card['names'])
    else:
      name = card['name']
    card['text'] = name + re.sub(r'\b' + re.escape(name) + r'\b', '~',
                                 card.get('text', ''))
    cards[name] = card
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
    line = line.split(' // ')[0]
    try:
      first_token, rest = line.split(maxsplit=1)
    except ValueError:
      continue
    if first_token.isnumeric():
      yield from [rest] * int(first_token)
    else:
      yield line


def CubeDiff(card_data, list_a, list_b):
  """Yield a diff between lists by linear sum assignment."""
  set_a = collections.Counter(ExpandList(list_a))
  set_b = collections.Counter(ExpandList(list_b))
  removes = list((set_a - set_b).elements())
  adds = list((set_b - set_a).elements())

  n, m = len(removes), len(adds)
  costs = numpy.zeros((n, m))
  for i in range(n):
    for j in range(m):
      costs[i, j] = CardDistance(card_data[removes[i]], card_data[adds[j]])
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
      yield f'+ {add}'


def main(argv):
  card_data = GetCards()
  list_a = [line.strip() for line in open(argv[1]).readlines()]
  list_b = [line.strip() for line in open(argv[2]).readlines()]

  def SortKey(change):
    card_a, card_b = change
    base = ('',)
    if not card_a or not card_b:
      base = ('___',)
    if not card_a:
      card_a = card_b
    colors = card_data[card_a]['colorIdentity']
    return base + (
        len(colors),
        tuple(WUBRG.index(c) for c in colors),
        int(card_data[card_a]['convertedManaCost']),
        card_a,
    )

  diff = list(CubeDiff(card_data, list_a, list_b))
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
