"""Diffing algorithm for Magic: the Gathering lists."""

from typing import List, Iterable, Tuple, Optional
import collections
import glob
import math
import pdb
import pickle
import re
import sys
import traceback

import numpy as np
import scipy.optimize

import color_distance
import constants
import html_utils
import mana_cost_distance
import oracle as oracle_lib
import types_distance

WEIGHTS = np.array([3, 1, 6, 2, 2])


def Metrics(tfidf_sq, a: oracle_lib.Card, b: oracle_lib.Card):
  """A metric for difference between cards a and b."""
  color = color_distance.EditDistance(a['colors'], b['colors'])
  color_identity = color_distance.EditDistance(a['color_identity'],
                                               b['color_identity'])
  mana_cost = mana_cost_distance.EditDistance(
      a.get('mana_cost', ''), b.get('mana_cost', ''))
  text_product = tfidf_sq[a['index'], b['index']]
  text = 1 - text_product
  types = types_distance.BucketDistance(a['type_line'], b['type_line'])

  metrics = np.array([color, color_identity, mana_cost, text, types])
  return metrics


def CardDistance(tfidf_sq, a: oracle_lib.Card, b: oracle_lib.Card):
  """A metric for difference between cards a and b."""
  metrics = Metrics(tfidf_sq, a, b)
  return weights.dot(metrics.T**2)


class CubeDiff:

  def __init__(self, oracle, list_a: List[str], list_b: List[str]):
    self.oracle = oracle
    self.set_a = collections.Counter(
        oracle.Canonicalize(name) for name in oracle_lib.ExpandList(list_a))
    self.set_b = collections.Counter(
        oracle.Canonicalize(name) for name in oracle_lib.ExpandList(list_b))

    self.removes = [
        self.oracle.Get(cardname)
        for cardname in (self.set_a - self.set_b).elements()
    ]
    self.adds = [
        self.oracle.Get(cardname)
        for cardname in (self.set_b - self.set_a).elements()
    ]
    self.PopulateMetrics()

  def PopulateMetrics(self):
    n, m = len(self.removes), len(self.adds)
    self.metrics = np.empty((n, m, 5))
    self.costs = np.empty((n, m))
    for i in range(n):
      remove = self.removes[i]
      for j in range(m):
        add = self.adds[j]
        self.metrics[i, j] = Metrics(self.oracle.GetTfidfSq(), remove, add)
        self.costs[i, j] = WEIGHTS.dot(self.metrics[i, j].T)

  def RawDiff(self) -> Iterable[Tuple[Optional[int], Optional[int]]]:
    """Yield a diff between lists by linear sum assignment."""
    n, m = len(self.removes), len(self.adds)
    rows, cols = scipy.optimize.linear_sum_assignment(self.costs)
    diff = zip(rows, cols)
    for remove, add in diff:
      yield (remove, add)
    if n > m:
      for extra_remove in set(range(n)) - set(rows):
        yield (extra_remove, None)
    if n < m:
      for extra_add in set(range(m)) - set(cols):
        yield (None, extra_add)

  def PageDiff(self):
    """Generate an HTML diff."""
    index_diff = sorted(self.RawDiff(), key=self._SortKey)
    card_diff = [(self.removes[r] if r is not None else None,
                  self.adds[a] if a is not None else None)
                 for r, a in index_diff]
    imagery = html_utils.GetImagery(self.oracle)

    yield '<html>'
    yield f'<head><style>{html_utils.CSS}</style>'
    yield '<link rel="icon" src="icon.png"></head>'
    yield '<body><ul>'
    for remove, add in card_diff:
      yield '<li class="change">'
      if remove and add:
        icon = '<img class="change-icon" src="Change.png" alt="becomes">'
      elif add:
        icon = '<img class="change-icon" src="Plus.png" alt="added">'
      elif remove:
        icon = '<img class="change-icon" src="Minus.png" alt="removed">'

      yield html_utils.CardImg(imagery, remove or 'ADDED')
      yield icon
      yield html_utils.CardImg(imagery, add or 'REMOVED')
      yield '</li>'
    yield '</ul></body></html>'

  def TextDiff(self):
    index_diff = sorted(self.RawDiff(), key=self._SortKey)
    card_diff = [(self.removes[r] if r is not None else None,
                  self.adds[a] if a is not None else None)
                 for r, a in index_diff]
    width_removes = max((len(r.shortname) for r, a in card_diff if r),
                        default=0)
    width_adds = max((len(a.shortname) for r, a in card_diff if a), default=0)
    for r, a in index_diff:
      remove = self.removes[r].shortname if r is not None else None
      add = self.adds[a].shortname if a is not None else None
      if remove and add:
        # yield ', '.join(f'{m:3.1f}' for m in self.metrics[r, a])
        yield f'  {remove:{width_removes}} -> {add:{width_adds}}'
      elif remove:
        yield f'- {remove}'
      else:
        yield f'{"":{width_removes}}  + {add}'

  def _SortKey(self, change):
    index_a, index_b = change
    card_a = self.removes[index_a] if index_a is not None else None
    card_b = self.adds[index_b] if index_b is not None else None
    if not card_a:
      card_a = card_b
    colors = card_a['colors']
    ci = card_a['color_identity']
    return (
        len(colors),
        sorted(tuple(constants.WUBRG.index(c) for c in colors)),
        sorted(tuple(constants.WUBRG.index(c) for c in ci)),
        int(card_a['cmc']),
        card_a['name'],
    )


def main(argv):
  list_a = [
      line.strip() for line in oracle_lib.ExpandList(open(argv[1]).readlines())
  ]
  list_b = [
      line.strip() for line in oracle_lib.ExpandList(open(argv[2]).readlines())
  ]
  oracle = oracle_lib.GetMaxOracle()
  cube_diff = CubeDiff(oracle, list_a, list_b)
  for line in cube_diff.PageDiff():
    print(line)


if __name__ == '__main__':
  try:
    main(sys.argv)
  except Exception as e:
    traceback.print_tb(e.__traceback__)
    print(repr(e))
    pdb.post_mortem()
    raise
