def EditDistance(a: str, b: str):
  """The edit distance between two color strings."""
  return len(set(a).symmetric_difference(set(b)))
