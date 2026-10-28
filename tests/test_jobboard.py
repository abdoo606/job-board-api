from __future__ import annotations

import pytest

from jobboard.search import Job, JobIndex, Query, tokenise

JOBS = [
    Job(1, "Senior Python Engineer", "Acme", "Berlin", True, 70000, 95000, ["python", "fastapi"], "Build APIs", 2),
    Job(2, "Data Engineer", "Globex", "Berlin", False, 60000, 80000, ["python", "sql"], "ETL pipelines", 10),
    Job(3, "Frontend Developer", "Initech", "Remote", True, 50000, 70000, ["react", "typescript"], "UI work", 1),
    Job(4, "Python Team Lead", "Acme", "Amsterdam", False, 90000, 120000, ["python", "leadership"], "Lead team", 30),
]


@pytest.fixture()
def index():
    return JobIndex(JOBS)


def test_tokenise_drops_stopwords_and_keeps_symbols():
    assert tokenise("The C++ and C# Developer") == ["c++", "c#", "developer"]


def test_search_ranks_title_match_first(index):
    results, _ = index.search(Query(q="python"))
    assert results[0].title in {"Senior Python Engineer", "Python Team Lead"}
    assert len(results) == 3


def test_search_is_case_insensitive(index):
    assert len(index.search(Query(q="PYTHON"))[0]) == 3


def test_no_terms_returns_everything(index):
    assert len(index.search(Query())[0]) == 4


def test_filters_combine(index):
    results, _ = index.search(Query(location="Berlin", remote=False))
    assert [j.id for j in results] == [2]


def test_remote_filter(index):
    results, _ = index.search(Query(remote=True))
    assert {j.id for j in results} == {1, 3}


def test_min_salary_uses_top_of_range(index):
    results, _ = index.search(Query(min_salary=100_000))
    assert [j.id for j in results] == [4]


def test_tag_filter_requires_all_tags(index):
    assert len(index.search(Query(tags=["python", "sql"]))[0]) == 1


def test_sort_newest(index):
    results, _ = index.search(Query(sort="newest"))
    assert [j.posted_days_ago for j in results] == sorted(j.posted_days_ago for j in results)


def test_sort_salary(index):
    results, _ = index.search(Query(sort="salary"))
    assert results[0].salary_max == 120000


def test_facets_reflect_the_filtered_set(index):
    _, facets = index.search(Query(location="Berlin"))
    assert facets["location"] == {"Berlin": 2}
    assert facets["remote"] == {"remote": 1, "onsite": 1}


def test_facet_tags_aggregate(index):
    _, facets = index.search(Query())
    assert facets["tags"]["python"] == 3


def test_unmatched_term_returns_nothing(index):
    assert index.search(Query(q="cobol"))[0] == []


def test_idf_downweights_common_terms(index):
    common = index._idf("python")
    rare = index._idf("leadership")
    assert rare > common
