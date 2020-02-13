"""Track cube changes over time."""
import collections
import datetime
import itertools
import os
import pprint

import bs4


def GetChanges():
  """Extract changes from CubeTutor blog html files."""
  results = []
  for name in os.listdir('ctdata'):
    if name.endswith('.html'):
      soup = bs4.BeautifulSoup(open(f'ctdata/{name}'), 'html.parser')
      for mode in ['ADD', 'REMOVE', 'SIDEBOARD', 'ADD_TO_SIDEBOARD']:
        for a in soup.select(f'.{mode}Icon'):
          stamp = datetime.datetime.strptime(
              a.parent.select('.date')[0].text, '%Y-%m-%d %H:%M')
          card = a.select('a')[0].text
          results.append((stamp, mode, card))
      mode = 'REPLACE'
      for a in soup.select(f'.{mode}Icon'):
        stamp = datetime.datetime.strptime(
            a.parent.select('.date')[0].text, '%Y-%m-%d %H:%M')
        card_out = a.select('a')[0].text
        card_in = a.select('a')[1].text
        results.append((stamp, 'REMOVE', card_out))
        results.append((stamp, 'ADD', card_in))
  return sorted(results, reverse=True)


# pprint.pprint(GetChanges(), stream=open('changes.txt', 'w'))


def TrackCube():
  """Construct the intermediate lists."""
  all_lists = {}
  counter = collections.Counter(
      line.strip().split(' // ')[0] for line in open('ChrissVintageCube.txt'))
  changes = GetChanges()
  for stamp, g in itertools.groupby(changes, key=lambda c: c[0]):
    all_lists[stamp] = counter.copy()
    for change in g:
      _, mode, card = change
      card = card.split(' // ')[0]
      # We're going backwards from the most recent list; operations are reversed
      # from their natural descriptions.
      if mode in ('SIDEBOARD', 'REMOVE'):
        counter.update([card])
      elif mode == 'ADD':
        counter.subtract([card])
      elif mode == 'ADD_TO_SIDEBOARD':
        pass
      else:
        raise ValueError('Unknown mode.')
  all_lists[datetime.datetime.fromtimestamp(0)] = counter.copy()
  return all_lists


x = TrackCube()
import IPython
IPython.embed()
