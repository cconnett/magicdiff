from typing import Iterable, Set


def TypeBucket(type_line: str) -> Iterable[str]:
  if 'Land' in type_line:
    yield 'Land'
  if 'Creature' in type_line:
    yield 'Creature'
  if 'Instant' in type_line:
    yield 'Instant'
    yield 'Sorcery'
  if 'Sorcery' in type_line:
    yield 'Sorcery'
  if 'Planeswalker' in type_line:
    yield 'Planeswalker'
  if 'Enchantment' in type_line:
    yield 'Enchantment'
  if 'Artifact' in type_line:
    yield 'Artifact'


def BucketDistance(a: str, b: str) -> int:
  return len(set(TypeBucket(a)).symmetric_difference(TypeBucket(b)))
