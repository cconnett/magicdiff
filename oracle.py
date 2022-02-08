import pickle


def GetCards(filename):
  """Read all cards from {filename}."""
  try:
    return pickle.load(open(f'{filename}.pkl', 'rb'))
  except (IOError, EOFError):
    pass

  card_list = json.load(open(filename))
  card_map = {
      card['name']: card
      for card in card_list
      if card['set_type'] not in ('token', 'vanguard', 'memorabilia')
  }
  partial_names = {}
  counter = itertools.count()
  for card in card_map.values():
    if 'card_faces' in card:
      card['oracle_text'] = '\n'.join(
          face['oracle_text'] for face in card['card_faces'])
      if 'colors' not in card:
        card['colors'] = [
            c for c in constants.WUBRG
            if any(c in face['colors'] for face in card['card_faces'])
        ]
      if 'mana_cost' not in card:
        card['mana_cost'] = card['card_faces'][0]['mana_cost']
    if 'oracle_text' not in card:
      card['oracle_text'] = ''
    if ' // ' in card['name']:
      cardname_pattern = '|'.join(
          re.escape(part) for part in card['name'].split(' // '))
      for part in card['name'].split(' // '):
        partial_names[part] = card
    else:
      cardname_pattern = card['name']
    card['oracle_text'] = re.sub(fr'\b{re.escape(cardname_pattern)}\b',
                                 'CARDNAME', card['oracle_text'])
    card['oracle_text'] = REMINDER.sub('', card['oracle_text'])
    card['index'] = next(counter)
  card_map['Life // Death']['mana_cost'] = '{1}{B}'
  assert len(card_map) == next(counter)
  pickle.dump((card_map, partial_names), open(f'{filename}.pkl', 'wb'))
  return card_map, partial_names
