CSS = '''
li {
    display: flex;
    align-items: center;
    margin: 0 3em 2em 0;
}

ul {
    list-style-type: none;
    display: flex;
    flex-flow: row wrap;
    justify-content: space-between;
}

img.change-icon {
    margin: 10;
}
img.card {
    width: 146;
}
.card-block {
    position: relative;
}
.card-name {
    position: absolute;
    left: 11px;
    top: 7px;
    z-index: 1;
    color: transparent;
    font-size: 12px;
}
'''


def CardImg(imagery, card):
  if card == 'REMOVED':
    return '<img class="card" src="BurnCard.png">'
  elif card == 'ADDED':
    return '<img class="card" src="UnburnCard.png">'
  elif card.name in imagery:
    return ('<div class="card-block">'
            f'<div class="card-name">{card.name}</div>'
            f'<img class="card" src="{imagery[card.name]}" '
            f'alt="{card.name}" title="{card.name}">'
            '</div>')
  else:
    return name


def GetImagery(oracle):
  """Get the imagery dictionary."""
  imagery = {
      card['name']: card['image_uris']['small']
      for card in oracle.oracle.values()
      if 'image_uris' in card
  }
  imagery.update({
      card['name']: card['card_faces'][0]['image_uris']['small']
      for card in oracle.oracle.values()
      if 'card_faces' in card and 'image_uris' not in card
  })
  return imagery
