"""In-memory activation buffer with background prefetch for decorrelation.

Uses a dedicated CUDA stream + background thread so buffer refills overlap
with training compute on the default stream.
"""

import threading

import torch

from spalf.data.store import ActivationStore


class ActivationBuffer:
    """Activation buffer that serves shuffled batches with async prefetch."""

    def __init__(
        self,
        store: ActivationStore,
        # Default 2^20 (~1M tokens); overridden by config.buffer_size in training.
        buffer_size: int = 2**20,
    ) -> None:
        self.store = store
        self.buffer_size = buffer_size
        self._ptr = 0
        self._total_tokens_served = 0
        self._buffer = self._initial_fill()

        # Background prefetch state.
        self._refill_stream = torch.cuda.Stream()
        self._refill_event: torch.cuda.Event | None = None
        self._refill_thread: threading.Thread | None = None

    def _initial_fill(self) -> torch.Tensor:
        """Fill the buffer from the activation store."""
        chunks: list[torch.Tensor] = []
        total = 0
        while total < self.buffer_size:
            batch = self.store.next_batch()
            chunks.append(batch)
            total += batch.shape[0]
        return torch.cat(chunks, dim=0)[: self.buffer_size]

    def _refill_half_impl(self) -> None:
        """Replace half the buffer with fresh activations on a dedicated CUDA stream."""
        half = self.buffer_size // 2
        chunks: list[torch.Tensor] = []
        total = 0
        while total < half:
            batch = self.store.next_batch()
            chunks.append(batch)
            total += batch.shape[0]
        fresh = torch.cat(chunks, dim=0)[:half]

        with torch.cuda.stream(self._refill_stream):
            end = self._ptr + half
            if end <= self.buffer_size:
                self._buffer[self._ptr : end] = fresh
            else:
                first_part = self.buffer_size - self._ptr
                self._buffer[self._ptr :] = fresh[:first_part]
                self._buffer[: half - first_part] = fresh[first_part:]
        self._ptr = end % self.buffer_size
        self._refill_event = self._refill_stream.record_event()

    def _wait_for_refill(self) -> None:
        """Block until any in-flight background refill completes."""
        if self._refill_thread is not None:
            self._refill_thread.join()
            self._refill_thread = None
        if self._refill_event is not None:
            self._refill_event.synchronize()
            self._refill_event = None

    def next_batch(self, batch_size: int) -> torch.Tensor:
        """Return a shuffled batch of activations [batch_size, d_model]."""
        # Ensure any prior refill is complete before reading.
        self._wait_for_refill()

        indices = torch.randint(0, self.buffer_size, (batch_size,), device="cuda")
        batch = self._buffer[indices]
        self._total_tokens_served += batch_size

        if self._total_tokens_served % self.buffer_size < batch_size:
            # Launch refill in background thread so training can proceed.
            self._refill_thread = threading.Thread(target=self._refill_half_impl, daemon=True)
            self._refill_thread.start()

        return batch
