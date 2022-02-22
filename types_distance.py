import functools
from typing import Iterable, Set, Tuple


@functools.cache
def TypeBucket(type_line: str) -> Tuple[str]:
  ret = []
  if 'Land' in type_line:
    ret.append('Land')
  if 'Creature' in type_line:
    ret.append('Creature')
  if 'Instant' in type_line:
    ret.append('Instant')
    ret.append('Sorcery')
  if 'Sorcery' in type_line:
    ret.append('Sorcery')
  if 'Planeswalker' in type_line:
    ret.append('Planeswalker')
  if 'Enchantment' in type_line:
    ret.append('Enchantment')
  if 'Artifact' in type_line:
    ret.append('Artifact')
  return tuple(ret)


def BucketDistance(a: str, b: str) -> int:
  return len(set(TypeBucket(a)).symmetric_difference(TypeBucket(b)))
