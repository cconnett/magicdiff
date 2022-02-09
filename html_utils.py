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
'''


def CardImg(imagery, name):
  if name == 'REMOVED':
    return '<img class="card" src="BurnCard.png">'
  elif name == 'ADDED':
    return '<img class="card" src="UnburnCard.png">'
  elif name in imagery:
    return f'<img class="card" src="{imagery[name]}">'
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
