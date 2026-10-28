# 15 · jobboard

Job board search with TF-IDF ranking and live facets.

## Run

```bash
from jobboard.search import JobIndex, Query
```

## Test

```bash
pytest projects/15-job-board -q
```

## What it demonstrates

- TF-IDF scoring with a title-match boost, plus idf down-weighting of common terms
- Facets are computed on the filtered set, not the whole corpus
- Filters compose: location, remote, salary ceiling, required tags
- Relevance, newest and salary sort modes with stable tie-breaking
