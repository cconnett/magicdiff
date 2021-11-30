# magicdiff

An algorithm for diffing Magic: The Gathering lists (decks, cubes, etc.)

## Setup

### Install required libraries

`pip install -r requirements.txt`

### Download Oracle data

Visit https://scryfall.com/docs/api/bulk-data and download the "Oracle Cards"
file into to this program's folder. The program will automatically read the
matching file with the latest timestamp.

## Running

`python diff.py list_a.txt list_b.txt > output.html`

Two filenames containing card lists are passed as arguments, and the HTML diff
report is written to standard out. Pipe that to the desired output file.

The lists should be simple text files containing one card name per line,
optionally with a leading numeric multiplicity.

Example:

```
  Borrowing 100,000 Arrows
  2 Ajani's Pridemate
```

is interpreted as

```
  Borrowing 100,000 Arrows
  Ajani's Pridemate
  Ajani's Pridemate
```
