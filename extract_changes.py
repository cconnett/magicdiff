import datetime
import os

import bs4


def GetChanges():
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


import pprint
pprint.pprint(GetChanges())
