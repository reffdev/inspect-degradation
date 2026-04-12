"""Change-point detection tests."""

from __future__ import annotations

import pytest

from inspect_degradation.analysis.change_point import (
    ChangePointResult,
    naive_change_point,
    pelt_change_points,
)


class TestNaive:
    def test_too_short_returns_none(self):
        assert naive_change_point([0, 1, 0]) is None

    def test_clean_step_recovered(self):
        series = [0.0] * 10 + [1.0] * 10
        idx = naive_change_point(series)
        assert idx == 10


class TestPELT:
    def test_short_series_returns_single_segment(self):
        result = pelt_change_points([0.0, 1.0])
        assert isinstance(result, ChangePointResult)
        assert result.change_points == []
        assert len(result.segment_means) == 1
        assert result.segments() == [(0, 2)]


# All other PELT tests require the optional `ruptures` dependency.
ruptures = pytest.importorskip("ruptures")


class TestPELTWithRuptures:
    def test_recovers_single_step(self):
        series = [0.0] * 15 + [1.0] * 15
        result = pelt_change_points(series, min_size=3)
        assert 15 in result.change_points
        assert len(result.segment_means) == len(result.change_points) + 1
        assert result.segment_means[0] == pytest.approx(0.0, abs=1e-9)
        assert result.segment_means[-1] == pytest.approx(1.0, abs=1e-9)

    def test_recovers_two_changes(self):
        series = [0.0] * 12 + [1.0] * 12 + [0.0] * 12
        result = pelt_change_points(series, min_size=3)
        # Should find both transitions; tolerate ±1 step from PELT cost ties.
        assert any(abs(cp - 12) <= 1 for cp in result.change_points)
        assert any(abs(cp - 24) <= 1 for cp in result.change_points)

    def test_high_penalty_returns_no_changes(self):
        series = [0.0] * 10 + [1.0] * 10
        result = pelt_change_points(series, penalty=10_000.0, min_size=3)
        assert result.change_points == []
        assert len(result.segment_means) == 1

    def test_segments_iterator_covers_full_series(self):
        series = [0.0] * 8 + [1.0] * 8
        result = pelt_change_points(series, min_size=2)
        segs = result.segments()
        assert segs[0][0] == 0
        assert segs[-1][1] == len(series)
        # Half-open intervals must tile contiguously.
        for i in range(len(segs) - 1):
            assert segs[i][1] == segs[i + 1][0]

    def test_to_dict_is_json_safe(self):
        import json

        result = pelt_change_points([0.0] * 10 + [1.0] * 10, min_size=2)
        json.dumps(result.to_dict())
