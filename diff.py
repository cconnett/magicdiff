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

import color_distance
import constants
import mana_cost_distance
import oracle
import types_distance

REMINDER = re.compile(r'\(.*\)')

ORACLE, PARTIALS = None, None


def Metrics(tfidf_sq, a, b):
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


def CardDistance(tfidf_sq, a, b):
  """A metric for difference between cards a and b."""
  color, color_identity, mana_cost, text, types = Metrics(tfidf_sq, a, b)

  weights = np.array([1, 2, 3, 1.4, 0.6])
  metrics = Metrics(tfidf_sq, a, b)
  return weights.dot(metrics.T**2)


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
  set_a = collections.Counter(oracle.ExpandList(list_a))
  set_b = collections.Counter(oracle.ExpandList(list_b))
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
  ORACLE, PARTIALS = oracle.GetMaxOracle()
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
      for line in oracle.ExpandList(open(argv[1]).readlines())
  ]
  list_b = [
      Canonicalize(line.strip())
      for line in oracle.ExpandList(open(argv[2]).readlines())
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
