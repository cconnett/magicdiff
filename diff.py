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
    name = card['name']
    card['text'] = name + re.sub(f'\\b{name}\\b', '~', card['text'])
    cards[name] = card
  return cards


def CubeDiff(card_data, list_a, list_b):
  set_a = collections.Counter(list_a)
  set_b = collections.Counter(list_b)
  removes = list(set_a - set_b)
  adds = list(set_b - set_a)

  n, m = len(removes), len(adds)
  costs = numpy.zeros((n, m))
  for i in range(n):
    for j in range(m):
      costs[i, j] = CardDistance(card_data[removes[i]], card_data[adds[j]])
  rows, cols = scipy.optimize.linear_sum_assignment(costs)
  solution = zip(rows, cols)
  width_removes = max(len(r) for r in removes)
  width_adds = max(len(a) for a in adds)
  for remove, add in solution:
    f = CardDistanceFeatures(card_data[removes[remove]], card_data[adds[add]])
    yield f'{removes[remove]:{width_removes}} -> {adds[add]:{width_adds}} '
    # f'[{f[0]:0.2f} {f[1]:0.2f} {f[2]:0.2f} {f[3]:0.2f}]'
  if n > m:
    for extra_remove in set(range(n)) - set(rows):
      yield f'-{removes[extra_remove]}'
  if n < m:
    for extra_add in set(range(m)) - set(cols):
      yield f'+{adds[extra_add]}'


def main():
  card_data = GetCards()
  list_a = [line.strip() for line in open(sys.argv[1]).readlines()]
  list_b = [line.strip() for line in open(sys.argv[2]).readlines()]
  lines = CubeDiff(card_data, list_a, list_b)
  for line in sorted(
      lines, key=lambda line: (line.startswith(('-', '+')), line)):
    print(line)


def test():
  card_data = GetCards()
  CubeDiff(card_data, [
      'Awakening Zone',
      'Grave Titan',
      'Lightning Bolt',
  ], [
      'Search for Tomorrow',
      'Awakening Zone',
      'Plaguecrafter',
  ])


if __name__ == '__main__':
  try:
    main()
  except:
    import pdb
    pdb.post_mortem()
