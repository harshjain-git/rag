"""
Unit tests for Gemini API rate-limit handling, retry logic, exponential backoff,
daily quota exhaustion, and permanent error propagation in infrastructure/gemini.py.
"""

import unittest
from unittest.mock import MagicMock, patch

from infrastructure.gemini import (
    GeminiLLM,
    GeminiDailyQuotaExhausted,
    is_daily_quota_error,
    is_retryable_gemini_error,
    extract_retry_delay,
)


class DummyAPIError(Exception):
    """Mock error simulating Google GenAI APIError with HTTP code."""
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class TestGeminiRetryLogic(unittest.TestCase):
    """Test suite verifying retry behavior, backoff, and quota guards in GeminiLLM."""

    def setUp(self):
        self.llm = GeminiLLM(api_key="test-key", model_name="test-model")
        # Speed up tests by reducing delays
        self.llm.DEFAULT_MAX_RETRIES = 3
        self.llm.DEFAULT_INITIAL_DELAY = 0.01
        self.llm.DEFAULT_BACKOFF_FACTOR = 1.5
        self.llm.MAX_DELAY = 1.0

    # 1. Detection of Daily Quota Exhaustion
    def test_daily_quota_detection(self):
        err_rpd = Exception("ResourceExhausted: Quota exceeded for quota metric 'requests' and limit 'Requests per day'")
        err_daily = Exception("429 Resource has been exhausted (e.g. check daily quota).")
        err_per_day = Exception("You exceeded your free_tier_requests_per_day quota.")
        err_rpm = Exception("429 Resource has been exhausted (e.g. check quota per minute).")

        self.assertTrue(is_daily_quota_error(err_rpd))
        self.assertTrue(is_daily_quota_error(err_daily))
        self.assertTrue(is_daily_quota_error(err_per_day))
        self.assertFalse(is_daily_quota_error(err_rpm))

    # 2. Detection of Retryable 429 and 503 Errors
    def test_retryable_error_detection(self):
        err_429 = DummyAPIError(429, "Too Many Requests")
        err_503 = DummyAPIError(503, "Service Unavailable")
        err_msg_429 = Exception("Rate limit reached. Please retry in 5s.")
        err_overloaded = Exception("Model is overloaded. Please try again later.")
        err_400 = DummyAPIError(400, "Bad Request: Invalid argument")
        err_401 = DummyAPIError(401, "Invalid API Key")
        err_rpd = Exception("Daily quota exceeded: free_tier_requests_per_day")

        self.assertTrue(is_retryable_gemini_error(err_429))
        self.assertTrue(is_retryable_gemini_error(err_503))
        self.assertTrue(is_retryable_gemini_error(err_msg_429))
        self.assertTrue(is_retryable_gemini_error(err_overloaded))
        self.assertFalse(is_retryable_gemini_error(err_400))
        self.assertFalse(is_retryable_gemini_error(err_401))
        # Daily quota must NOT be marked retryable
        self.assertFalse(is_retryable_gemini_error(err_rpd))

    # 3. Server Retry Delay Extraction
    def test_extract_retry_delay(self):
        err_with_seconds = Exception("Quota exceeded. Please retry in 7.5 seconds.")
        err_with_s = Exception("Rate limit: retry after 12s.")
        err_without_delay = Exception("429 Too Many Requests")

        # Returns parsed duration + 1.0s buffer
        self.assertAlmostEqual(extract_retry_delay(err_with_seconds), 8.5)
        self.assertAlmostEqual(extract_retry_delay(err_with_s), 13.0)
        self.assertEqual(extract_retry_delay(err_without_delay, fallback=2.0), 2.0)

    # 4. Immediate Fast-Fail on Daily Quota (No Retries Attempted)
    def test_daily_quota_fails_immediately_without_retries(self):
        mock_func = MagicMock(side_effect=Exception("Daily quota exhausted: requests per day limit reached"))

        with self.assertRaises(GeminiDailyQuotaExhausted):
            self.llm._call_with_retry(mock_func)

        # Must have been called exactly ONCE: zero retries wasted
        self.assertEqual(mock_func.call_count, 1)

    # 5. Permanent Errors (400, 401, etc.) Propagated Immediately
    def test_permanent_errors_not_retried(self):
        mock_func = MagicMock(side_effect=DummyAPIError(400, "Invalid parameter"))

        with self.assertRaises(DummyAPIError):
            self.llm._call_with_retry(mock_func)

        self.assertEqual(mock_func.call_count, 1)

    # 6. Retry on Temporary 429 and Eventual Success
    @patch("time.sleep", return_value=None)
    def test_retry_on_429_success(self, mock_sleep):
        mock_response = MagicMock(text='{"status": "ok"}')
        mock_func = MagicMock(side_effect=[
            DummyAPIError(429, "Rate limit reached. Please retry in 2s."),
            mock_response,
        ])

        result = self.llm._call_with_retry(mock_func)
        self.assertEqual(result, mock_response)
        self.assertEqual(mock_func.call_count, 2)
        mock_sleep.assert_called_once()

    # 7. Retry on Temporary 503 and Eventual Success
    @patch("time.sleep", return_value=None)
    def test_retry_on_503_success(self, mock_sleep):
        mock_response = MagicMock(text="Hello world")
        mock_func = MagicMock(side_effect=[
            DummyAPIError(503, "Service Unavailable: Backend overloaded"),
            mock_response,
        ])

        result = self.llm._call_with_retry(mock_func)
        self.assertEqual(result, mock_response)
        self.assertEqual(mock_func.call_count, 2)
        mock_sleep.assert_called_once()

    # 8. Retry Limit Exhaustion Raises Final Error
    @patch("time.sleep", return_value=None)
    def test_retry_exhaustion_raises(self, mock_sleep):
        mock_func = MagicMock(side_effect=DummyAPIError(429, "Rate limit exceeded"))

        with self.assertRaises(DummyAPIError):
            self.llm._call_with_retry(mock_func)

        # 1 initial try + 3 retries = 4 total attempts
        self.assertEqual(mock_func.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    # 9. Consistent Application to generate_text, generate_json, generate_with_tools
    @patch("time.sleep", return_value=None)
    def test_methods_use_retry_wrapper(self, mock_sleep):
        mock_gen_content = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_content = mock_gen_content
        self.llm._client = mock_client

        # A: generate_text with 1 retry
        mock_gen_content.side_effect = [
            DummyAPIError(429, "Rate limit"),
            MagicMock(text="Answer text"),
        ]
        text_res = self.llm.generate_text(contents="test")
        self.assertEqual(text_res, "Answer text")
        self.assertEqual(mock_gen_content.call_count, 2)

        # B: generate_json with 1 retry
        mock_gen_content.side_effect = [
            DummyAPIError(503, "Unavailable"),
            MagicMock(text='{"key": "value"}'),
        ]
        json_res = self.llm.generate_json(contents="test")
        self.assertEqual(json_res, {"key": "value"})
        self.assertEqual(mock_gen_content.call_count, 4)

        # C: generate_with_tools with 1 retry
        mock_response = MagicMock(text="Tool result", function_calls=None)
        mock_gen_content.side_effect = [
            DummyAPIError(429, "Rate limit"),
            mock_response,
        ]
        tool_res = self.llm.generate_with_tools(contents="test", tools=[])
        self.assertEqual(tool_res["call_type"], "text")
        self.assertEqual(tool_res["text"], "Tool result")
        self.assertEqual(mock_gen_content.call_count, 6)


if __name__ == "__main__":
    unittest.main()
