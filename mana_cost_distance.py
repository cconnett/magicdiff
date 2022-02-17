import collections
import functools
import re

import numpy as np

import constants

PIP = re.compile(r'\{(.*?)\}')
HYBRID = re.compile('([2WUBRG])/([WUBRGP])')
HYBRID_PHYREXIAN = re.compile('([WUBRG])/([WUBRG]/P)')


@functools.cache
def FlattenManaCost(mana_cost: str):
  """Reduce a mana cost down to array: W pips, U, B, R, G, ~mana value."""
  mana_cost = ''.join(mana_cost.split(' // '))
  accumulator = collections.Counter()
  pips = PIP.findall(mana_cost)
  for p in pips:
    if p in constants.WUBRG:
      accumulator[p] += 1
      accumulator['V'] += 1
    elif h := HYBRID.fullmatch(p):
      left, right = h.groups()
      if right == 'P':
        accumulator[left] += 1 / 3
        accumulator['V'] += 1 / 3
      elif left == '2':
        accumulator[right] += 2 / 3
        accumulator['V'] += 4 / 3
      else:
        accumulator[left] += 0.5
        accumulator[right] += 0.5
      accumulator['V'] += 1
    elif p in 'XYZ':
      accumulator['V'] += 3
    elif p in ('C', 'S'):  # Colorless cost; snow
      accumulator['V'] += 1  # Mana value accumulator
    elif p.startswith('H'):  # Half mana
      accumulator[p[1]] += 0.5
      accumulator['V'] += 0.5
    elif h := HYBRID_PHYREXIAN.fullmatch(p):
      left, right = h.groups()
      accumulator[left] += 1 / 6
      accumulator[right] += 1 / 6
      accumulator['V'] += 1 / 3
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


def EditDistance(mana_cost_a: str, mana_cost_b: str):
  """Distance between two mana costs by edit distance."""
  return sum(abs(FlattenManaCost(mana_cost_a) - FlattenManaCost(mana_cost_b)))
