"""Embedding generation using OpenAI API."""
# type: ignore  # Legacy file - uses OpenAI which has import issues

import asyncio
import logging
from typing import List

import openai
from asyncio_throttle import Throttler

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages OpenAI embedding generation with rate limiting."""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        # Rate limit: 3000 requests per minute for OpenAI API
        self.throttler = Throttler(rate_limit=50, period=1)  # 50 per second

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * 1536  # Return zero vector for empty text

        # Truncate text if too long (OpenAI has 8191 token limit)
        if len(text) > 8000:
            text = text[:8000] + "..."
            logger.debug("Truncated long text for embedding")

        async with self.throttler:
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=text.replace("\n", " "),  # Clean newlines
                )
                return response.data[0].embedding

            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel."""
        if not texts:
            return []

        # Process in smaller batches to respect rate limits
        batch_size = 20
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.generate_embedding(text) for text in batch], return_exceptions=True
            )

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to generate embedding for text {i + j}: {result}")
                    results.append([0.0] * 1536)  # Zero vector fallback
                else:
                    results.append(result)

            # Small delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        return results

    def format_commit_for_embedding(self, message: str, diff: str) -> str:
        """Format commit message and diff into text suitable for embedding."""
        # Combine message and diff with clear separation
        formatted_text = f"Commit message: {message.strip()}\n\n"

        if diff.strip():
            # Truncate diff if too long
            max_diff_len = 6000  # Leave room for message
            if len(diff) > max_diff_len:
                diff = diff[:max_diff_len] + "\n... (truncated)"

            formatted_text += f"Code changes:\n{diff}"

        return formatted_text
