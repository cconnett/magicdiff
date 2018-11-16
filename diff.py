from typing import List

import collections
import difflib
import json
import re
import sys

import numpy
import scipy.optimize


def ColorDistance(a: List[str], b: List[str]) -> float:
  a, b = set(a), set(b)
  if not a and a == b:
    return 0
  return (1 - (2 * len(a & b) / (len(a) + len(b))))


def TextDistance(a: str, b: str) -> float:
  return 1 - difflib.SequenceMatcher(a=a, b=b).ratio()


def TypeBucket(types: List[str]) -> str:
  if 'Land' in types:
    return 'Land'
  if 'Creature' in types:
    return 'Creature'
  if 'Instant' in types or 'Sorcery' in types:
    return 'Spell'
  return 'Permanent'


def TypesDistance(a: List[str], b: List[str]) -> int:
  return 1 - bool(TypeBucket(a) == TypeBucket(b))


def CmcMetric(card):
  return card['convertedManaCost'] + (4 if '{X}' in card.get('manaCost', '')
                                      else 0)


def CardDistanceFeatures(a, b):
  color = ColorDistance(a['colors'], b['colors'])
  text = TextDistance(a['text'], b['text'])
  types = TypesDistance(a['types'], b['types'])
  cmc = min(4, abs(CmcMetric(a) - CmcMetric(b))) / 4

  return [4 * color, 2 * text, 0.3 * types, 1.5 * cmc]


def CardDistance(a, b):
  features = CardDistanceFeatures(a, b)
  return sum(features) / len(features)


def GetCards():
  c = json.load(open('AllCards.json'))
  cards = {}
  for key, card in c.items():
    if card['layout'] == 'split':
      name = ' // '.join(card['names'])
    else:
      name = card['name']
    card['text'] = name + re.sub(f'\\b{name}\\b', '~', card['text'])
    cards[name] = card
  return cards


def ExpandList(lst):
  for line in lst:
    line = line.strip()
    try:
      first_token, rest = line.split(maxsplit=1)
    except ValueError:
      continue
    if first_token.isnumeric():
      yield from [rest] * int(first_token)
    else:
      yield line


def CubeDiff(card_data, list_a, list_b):
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
  width_removes = max(len(r) for r in removes)
  width_adds = max(len(a) for a in adds)
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
  diff = sorted(
      CubeDiff(card_data, list_a, list_b),
      key=lambda entry: (-all(entry), entry[0]))
  for line in FormatDiff(diff):
    print(line)


def test():
  main(['diff.py', 'removes.txt', 'adds.txt'])


if __name__ == '__main__':
  try:
    main(sys.argv)
  except Exception as e:
    import traceback
    traceback.print_tb(e.__traceback__)
    print(e)
    import pdb
    pdb.post_mortem()
