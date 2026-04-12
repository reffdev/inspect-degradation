"""Drift canary fingerprint and comparison."""

from __future__ import annotations

from inspect_degradation.grader.drift_canary import (
    CanaryFingerprint,
    compare_fingerprints,
    fingerprint_from_response,
)


class TestFingerprint:
    def test_identical_responses_produce_identical_hashes(self):
        a = fingerprint_from_response(model="m", response="hello world\n")
        b = fingerprint_from_response(model="m", response="hello world")
        # Whitespace stripping means trailing newlines don't trip equality.
        assert a.response_sha256 == b.response_sha256

    def test_different_responses_diverge(self):
        a = fingerprint_from_response(model="m", response="hello")
        b = fingerprint_from_response(model="m", response="world")
        assert a.response_sha256 != b.response_sha256

    def test_excerpt_truncates_at_200(self):
        long = "x" * 500
        f = fingerprint_from_response(model="m", response=long)
        assert len(f.response_excerpt) == 200

    def test_round_trip_through_dict(self):
        a = fingerprint_from_response(model="m", response="hi")
        b = CanaryFingerprint.from_dict(a.to_dict())
        assert a.matches(b)

    def test_to_dict_is_json_safe(self):
        import json

        a = fingerprint_from_response(model="m", response="hi")
        json.dumps(a.to_dict())


class TestCompareFingerprints:
    def test_match_when_everything_agrees(self):
        a = fingerprint_from_response(model="m", response="hi")
        b = fingerprint_from_response(model="m", response="hi")
        result = compare_fingerprints(a, b)
        assert result.match is True
        assert "match" in result.notes

    def test_drift_detected_on_response_change(self):
        a = fingerprint_from_response(model="m", response="hi")
        b = fingerprint_from_response(model="m", response="HI")
        result = compare_fingerprints(a, b)
        assert result.match is False
        assert result.models_match is True
        assert result.prompts_match is True
        assert "DRIFT" in result.notes

    def test_model_name_change_detected(self):
        a = fingerprint_from_response(model="m1", response="hi")
        b = fingerprint_from_response(model="m2", response="hi")
        result = compare_fingerprints(a, b)
        assert result.match is False
        assert result.models_match is False
        assert "model name" in result.notes

    def test_prompt_change_detected(self):
        a = fingerprint_from_response(model="m", response="hi", prompt="P1")
        b = fingerprint_from_response(model="m", response="hi", prompt="P2")
        result = compare_fingerprints(a, b)
        assert result.match is False
        assert result.prompts_match is False
        assert "prompt" in result.notes

    def test_to_dict_is_json_safe(self):
        import json

        a = fingerprint_from_response(model="m", response="hi")
        b = fingerprint_from_response(model="m", response="bye")
        json.dumps(compare_fingerprints(a, b).to_dict())
