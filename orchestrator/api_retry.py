"""Retry wrapper for Anthropic API calls with exponential backoff."""
from __future__ import annotations

import time


def call_with_retry(client, max_retries: int = 3, **kwargs):
    """Call client.messages.create with retry on rate limit errors.

    Args:
        client: anthropic.Anthropic() client instance
        max_retries: Number of retries on 429/529 errors
        **kwargs: Passed to client.messages.create()

    Returns:
        The API response
    """
    import anthropic

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff: 15s, 30s, 60s
                wait = 15 * (2 ** attempt)
                status = getattr(e, 'status_code', '?')
                print(f"  ⏳ Rate limit ({status}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise last_error  # Should never reach here
