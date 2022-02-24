"""Diffing algorithm for Magic: the Gathering lists."""

from typing import List, Iterable, Tuple, Optional
import collections
import glob
import math
import multiprocessing
import pickle
import re
import sys
import time
import traceback

from absl import app
from absl import flags
import h5py
import numpy as np
import scipy.optimize
import yappi

import color_distance
import constants
import html_utils
import mana_cost_distance
import oracle as oracle_lib
import types_distance

FLAGS = flags.FLAGS
WEIGHTS = np.array([3, 1, 6, 2, 0.2]).T
COSTS_FILENAME = '/tmp/costs.hdf5'

flags.DEFINE_bool(
    'html', False, 'Produce an html diff rather than text.', short_name='g')
flags.DEFINE_bool('compute', False,
                  'Compute the costs matrix and write to disk.')


def Metrics(tfidf_sq, a: oracle_lib.Card, b: oracle_lib.Card):
  """A metric for difference between cards a and b."""
  a.Parse()
  b.Parse()
  color = color_distance.EditDistance(a['colors'], b['colors'])
  color_identity = color_distance.EditDistance(a['color_identity'],
                                               b['color_identity'])
  mana_cost = mana_cost_distance.EditDistance(
      a.get('mana_cost', ''), b.get('mana_cost', ''))
  text_product = tfidf_sq[a.index, b.index]
  text = 1 - text_product
  types = types_distance.BucketDistance(a['type_line'], b['type_line'])

  metrics = np.array([color, color_identity, mana_cost, text, types])
  return metrics


class MagicDiff:

  def __init__(self, oracle, list_a: List[oracle_lib.Card],
               list_b: List[oracle_lib.Card]):
    self.oracle = oracle
    self.global_costs = None
    set_a = collections.Counter(list_a)
    set_b = collections.Counter(list_b)
    self.removes = list((set_a - set_b).elements())
    self.adds = list((set_b - set_a).elements())

  def GetGlobalCosts(self):
    if self.global_costs is not None:
      return self.global_costs
    try:
      self._LoadGlobalCosts()
    except (IOError, KeyError):
      if FLAGS.compute:
        self._WriteGlobalCostsFile()
        self._LoadGlobalCosts()
    return self.global_costs

  def _WriteGlobalCostsFile(self):
    with h5py.File(COSTS_FILENAME, 'w') as f:
      costs = f.create_dataset(
          'costs', (len(self.oracle.oracle), len(self.oracle.oracle)),
          dtype='f4')
    self.oracle.GetTfidfSq()
    self.oracle.tfidf_sq = None
    with multiprocessing.Pool() as pool:
      for i, costs_i in enumerate(
          pool.imap(
              self._CostsFor,
              enumerate(self.oracle.oracle.values()),
              chunksize=70)):
        print(f'{i+1:05d} of {len(self.oracle.oracle)}')

  def _CostsFor(self, args):
    (i, c_i) = args
    costs_i = np.zeros((len(self.oracle.oracle),), dtype='f4')
    for j, c_j in enumerate(self.oracle.oracle.values()):
      if i < j:
        m = Metrics(self.oracle.GetTfidfSq(), c_i, c_j)
        costs_i[j] = m.dot(WEIGHTS)
    while True:
      try:
        with h5py.File(COSTS_FILENAME, 'a') as f:
          f['costs'][i] = costs_i
        break
      except BlockingIOError:
        time.sleep(0.1)

  def _LoadGlobalCosts(self):
    try:
      f = h5py.File(COSTS_FILENAME)
      self.global_costs = f['costs']
    except KeyError:
      f.close()
      self.global_costs = None
      raise

  def PopulateMetrics(self):
    n, m = len(self.removes), len(self.adds)
    self.costs = np.empty((n, m))
    for i in range(n):
      remove = self.removes[i]
      remove.Parse()
      for j in range(m):
        add = self.adds[j]
        add.Parse()
        if self.global_costs is None:
          self.costs[i, j] = Metrics(self.oracle.GetTfidfSq(), remove,
                                     add).dot(WEIGHTS)
        else:
          self.costs[i, j] = self.global_costs[min(remove.index, add.index),
                                               max(remove.index, add.index)]

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
                        default=1)
    width_adds = max((len(a.shortname) for r, a in card_diff if a), default=1)
    for r, a in index_diff:
      remove = self.removes[r].shortname if r is not None else None
      add = self.adds[a].shortname if a is not None else None
      if remove and add:
        # yield ', '.join(f'{m:3.1f}' for m in self.metrics[r, a])
        yield f'  {remove:{width_removes}} 🡺  {add:{width_adds}}'
      elif remove:
        yield f'- {remove}'
      else:
        yield f'{"":{width_removes}}   +  {add}'

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
  # yappi.start()
  s = time.time()
  oracle = oracle_lib.GetLiteOracle()
  print(f'Loaded oracle in {time.time() - s:.2f}s.', file=sys.stderr)

  list_a = oracle_lib.CardList(open(argv[1]).readlines(), oracle)
  list_b = oracle_lib.CardList(open(argv[2]).readlines(), oracle)

  incomplete_sections = set(list_a.sections.keys()).symmetric_difference(
      list_b.sections.keys())
  for section in list_a.sections:
    if section in list_b.sections:
      diff = MagicDiff(oracle, list_a.sections[section].cards,
                       list_b.sections[section].cards)
      diff.GetGlobalCosts()
      diff.PopulateMetrics()
      if section:
        print(f'### {section} ###')
      for line in diff.TextDiff():
        print(line)
  rest_a = []
  rest_b = []
  for section in incomplete_sections:
    if section in list_a.sections:
      rest_a.extend(list_a.sections[section].cards)
    if section in list_b.sections:
      rest_b.extend(list_b.sections[section].cards)
  diff = MagicDiff(oracle, rest_a, rest_b)
  print('Computing costs.', file=sys.stderr)
  s = time.time()
  diff.GetGlobalCosts()
  diff.PopulateMetrics()
  print(f'Computed costs in {time.time() - s:.2f}s.', file=sys.stderr)
  diff_lines = diff.PageDiff() if FLAGS.html else diff.TextDiff()
  for line in diff_lines:
    print(line)
  yappi.get_func_stats().print_all()


if __name__ == '__main__':
  try:
    app.run(main)
  except Exception as e:
    import pdb
    traceback.print_tb(e.__traceback__)
    print(repr(e))
    pdb.post_mortem()
    raise
