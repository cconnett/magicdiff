import itertools
import pickle
import re

import oracle as oracle_lib

KEYS_TO_DELETE = {
    'artist', 'artist_ids', 'booster', 'border_color', 'card_back_id',
    'cardmarket_id', 'collector_number', 'digital', 'edhrec_rank', 'finishes',
    'flavor_text', 'foil', 'frame', 'full_art', 'games', 'highres_image', 'id',
    'illustration_id', 'image_status', 'keywords', 'lang', 'layout',
    'legalities', 'mtgo_foil_id', 'mtgo_id', 'multiverse_ids', 'nonfoil',
    'object', 'oracle_id', 'oversized', 'prices', 'prints_search_uri', 'promo',
    'rarity', 'related_uris', 'released_at', 'reprint', 'reserved',
    'rulings_uri', 'scryfall_set_uri', 'scryfall_uri', 'set', 'set_id',
    'set_name', 'set_search_uri', 'set_uri', 'story_spotlight', 'tcgplayer_id',
    'textless', 'uri', 'variation'
}


def ReduceOracle(oracle):
  for card in oracle.oracle.values():
    card.Parse()
    for key in KEYS_TO_DELETE:
      card.json.pop(key, None)

  cards_to_delete = []
  for name, card in oracle.oracle.items():
    if re.search(r'\b(Plane|Scheme)\b', card.get('type_line', '')):
      cards_to_delete.append(name)
  for name in cards_to_delete:
    oracle.oracle.pop(name.lower())
    oracle.partials.pop(name.lower())
    oracle.all_names.remove(name.lower())

  counter = itertools.count()
  for card in oracle.oracle.values():
    card.index = next(counter)


if __name__ == '__main__':
  oracle = oracle_lib.GetMaxOracle()
  ReduceOracle(oracle)
  pickle.dump(oracle, open('lite-oracle.pkl', 'wb'))
