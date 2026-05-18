import sqlite3

from evalscope.perf.plugin.api.default_api import DefaultApiPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.db_util import create_result_table, get_percentile_results, insert_benchmark_data
from evalscope.perf.utils.perf_constants import Metrics


class _NoopApiPlugin:

    def parse_responses(self, *args: object, **kwargs: object) -> tuple[int, int]:
        raise AssertionError('Token counts should already be populated.')


def _make_benchmark_data(prompt_tokens: int, cached_tokens: int) -> BenchmarkData:
    data = BenchmarkData(
        success=True,
        start_time=1.0,
        completed_time=2.0,
        query_latency=1.0,
        first_chunk_latency=0.1,
        time_per_output_token=0.01,
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
        real_cached_tokens=cached_tokens,
    )
    data.finalize(_NoopApiPlugin())
    return data


def test_default_api_extracts_cached_tokens_from_usage() -> None:
    assert DefaultApiPlugin._extract_cached_tokens({
        'prompt_tokens': 2006,
        'prompt_tokens_details': {
            'cached_tokens': 1920,
        },
    }) == 1920
    assert DefaultApiPlugin._extract_cached_tokens({
        'prompt_tokens': 2006,
        'prompt_tokens_details': None,
    }) == 0


def test_summary_reports_average_prefix_cache_hit_rate() -> None:
    accumulator = MetricsAccumulator(concurrency=1, rate=-1)

    for data in [
        _make_benchmark_data(prompt_tokens=100, cached_tokens=50),
        _make_benchmark_data(prompt_tokens=100, cached_tokens=0),
        _make_benchmark_data(prompt_tokens=100, cached_tokens=100),
    ]:
        accumulator.update(data, _NoopApiPlugin())

    message = accumulator.to_result().create_message(api_type='openai')

    assert message[Metrics.AVERAGE_CACHED_PERCENT] == 50.0


def test_percentiles_report_prefix_cache_hit_rate(tmp_path) -> None:
    db_path = tmp_path / 'benchmark_data.db'
    with sqlite3.connect(db_path) as con:
        cursor = con.cursor()
        create_result_table(cursor)
        for data in [
            _make_benchmark_data(prompt_tokens=100, cached_tokens=50),
            _make_benchmark_data(prompt_tokens=100, cached_tokens=0),
            _make_benchmark_data(prompt_tokens=100, cached_tokens=100),
        ]:
            insert_benchmark_data(cursor, data)
        con.commit()

    percentile_result = get_percentile_results(str(db_path), api_type='openai')

    assert percentile_result.get_p('1%', 'prefix_cache_hit_rate') == 0
    assert percentile_result.get_p('50%', 'prefix_cache_hit_rate') == 50
    assert percentile_result.get_p('99%', 'prefix_cache_hit_rate') == 100
