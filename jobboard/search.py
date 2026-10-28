"""Search and faceting over an in-memory corpus - no external search engine needed."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN = re.compile(r"[a-z0-9+#.]+")
STOPWORDS = {"a", "an", "the", "and", "or", "for", "in", "of", "to", "with", "at", "on"}


def tokenise(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(slots=True)
class Job:
    id: int
    title: str
    company: str
    location: str
    remote: bool
    salary_min: int
    salary_max: int
    tags: list[str] = field(default_factory=list)
    description: str = ""
    posted_days_ago: int = 0

    @property
    def searchable(self) -> str:
        return f"{self.title} {self.company} {self.description} {' '.join(self.tags)}"


@dataclass(slots=True)
class Query:
    q: str | None = None
    location: str | None = None
    remote: bool | None = None
    min_salary: int | None = None
    tags: list[str] = field(default_factory=list)
    sort: str = "relevance"  # relevance | newest | salary


class JobIndex:
    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = {job.id: job for job in jobs}
        self._doc_freq: Counter[str] = Counter()
        self._tokens: dict[int, Counter[str]] = {}
        for job in jobs:
            counts = Counter(tokenise(job.searchable))
            self._tokens[job.id] = counts
            self._doc_freq.update(counts.keys())

    def _idf(self, term: str) -> float:
        n = len(self.jobs)
        df = self._doc_freq.get(term, 0)
        return math.log((n + 1) / (df + 1)) + 1

    def _score(self, job: Job, terms: list[str]) -> float:
        counts = self._tokens[job.id]
        score = 0.0
        title_tokens = set(tokenise(job.title))
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            score += (1 + math.log(tf)) * self._idf(term)
            if term in title_tokens:
                score *= 1.5  # a title match matters more than a body match
        return score

    def search(self, query: Query) -> tuple[list[Job], dict[str, dict[str, int]]]:
        terms = tokenise(query.q or "")
        results = []
        for job in self.jobs.values():
            if query.location and query.location.lower() not in job.location.lower():
                continue
            if query.remote is not None and job.remote is not query.remote:
                continue
            if query.min_salary is not None and job.salary_max < query.min_salary:
                continue
            if query.tags and not set(t.lower() for t in query.tags) <= {t.lower() for t in job.tags}:
                continue
            score = self._score(job, terms)
            if terms and score == 0:
                continue
            results.append((score, job))

        if query.sort == "newest":
            results.sort(key=lambda pair: pair[1].posted_days_ago)
        elif query.sort == "salary":
            results.sort(key=lambda pair: -pair[1].salary_max)
        else:
            results.sort(key=lambda pair: (-pair[0], pair[1].posted_days_ago))

        matched = [job for _, job in results]
        return matched, self.facets(matched)

    @staticmethod
    def facets(jobs: list[Job]) -> dict[str, dict[str, int]]:
        return {
            "location": dict(Counter(j.location for j in jobs).most_common()),
            "tags": dict(Counter(t for j in jobs for t in j.tags).most_common()),
            "remote": {"remote": sum(j.remote for j in jobs), "onsite": sum(not j.remote for j in jobs)},
        }
