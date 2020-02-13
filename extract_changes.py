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
  return results


def ConditionalZip(badges, cards):
  badges = iter(badges)
  cards = iter(cards)
  mapping = {'+': 'ADD', '-': 'REMOVE'}
  for badge in badges:
    if badge in ('+', '-'):
      yield (mapping[badge], next(cards))
    elif badge in ('→',):
      yield (mapping['-'], next(cards))
      yield (mapping['+'], next(cards))


def GetChangesCobra():
  """Extract changes from CubeCobra blog html files."""
  results = []
  for name in os.listdir('ccdata'):
    if name.endswith('.html'):
      soup = bs4.BeautifulSoup(open(f'ccdata/{name}'), 'html.parser')
      posts = soup.select('.mt-3')
      for post in posts:
        if 'Cube Bulk Import' in post.select('.card-title')[0].text:
          continue
        deltatext = soup.select('h6.mb-2')[0].text.split('-')[-1].strip()
        count, unit, unused_ago = deltatext.split()
        if not unit.endswith('s'):
          unit += 's'
        delta = datetime.timedelta(**{unit: int(count)})
        stamp = datetime.datetime.now() - delta
        badges = [span.text for span in post.select('span.badge')]
        cards = [a.text for a in post.select('a.dynamic-autocard')]
        results.extend(
            (stamp,) + change for change in ConditionalZip(badges, cards))
  return results


# pprint.pprint(GetChanges(), stream=open('changes.txt', 'w'))


def TrackCube():
  """Construct the intermediate lists."""
  all_lists = {}
  counter = collections.Counter(
      line.strip().split(' // ')[0] for line in open('ChrissVintageCube.txt'))
  changes = GetChanges() + GetChangesCobra()
  changes.sort(reverse=True)
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


try:
  x = TrackCube()
except:
  import pdb
  pdb.post_mortem()
import IPython
IPython.embed()
