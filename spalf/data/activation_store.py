"""Activation streaming via HuggingFace model hooks."""

from collections.abc import Iterator

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from spalf.constants import DEVICE


class ActivationStore:
    """Streams residual activations from HuggingFace causal language models."""

    def __init__(
        self,
        model_name: str,
        hook_point: str,
        dataset_name: str,
        batch_size: int,
        seq_len: int = 128,
        text_column: str = "text",
        dataset_split: str = "train",
        dataset_config: str = "",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.hook_point = hook_point
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.text_column = text_column
        self.dataset_split = dataset_split
        self.dataset_config = dataset_config
        self.seed = seed
        self.device = DEVICE

        self._hook_handle = None
        self._captured_activations: torch.Tensor | None = None
        self._token_iter: Iterator[torch.Tensor] | None = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self._hf_target_module = dict(self.model.named_modules())[self.hook_point]
        self._register_hf_hook(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _register_hf_hook(self, model: AutoModelForCausalLM) -> None:
        """Register a forward hook to capture residual stream activations."""

        def hook_fn(_module, _input, output):
            self._captured_activations = output[0].detach()

        self._hook_handle = self._hf_target_module.register_forward_hook(hook_fn)

    def _token_generator(self) -> Iterator[torch.Tensor]:
        """Yield batches of token IDs. Shuffled streaming with batched tokenization."""
        target_len = self.batch_size * self.seq_len
        tokenize_batch_size = 256
        shuffle_buffer_size = 10_000

        epoch = 0
        while True:
            load_kwargs = {"split": self.dataset_split, "streaming": True}
            if self.dataset_config:
                load_kwargs["name"] = self.dataset_config
            dataset = load_dataset(self.dataset_name, **load_kwargs)
            dataset = dataset.select_columns([self.text_column])
            dataset = dataset.shuffle(seed=self.seed, buffer_size=shuffle_buffer_size)
            dataset.set_epoch(epoch)

            token_buffer: list[int] = []
            text_batch: list[str] = []

            for example in dataset:
                text_batch.append(example[self.text_column])

                if len(text_batch) >= tokenize_batch_size:
                    encoded = self.tokenizer(
                        text_batch, add_special_tokens=False
                    )["input_ids"]
                    for ids in encoded:
                        token_buffer.extend(ids)
                    text_batch = []

                    while len(token_buffer) >= target_len:
                        batch_tokens = token_buffer[:target_len]
                        token_buffer = token_buffer[target_len:]
                        yield torch.tensor(batch_tokens, dtype=torch.long).reshape(
                            self.batch_size, self.seq_len
                        )

            if text_batch:
                encoded = self.tokenizer(
                    text_batch, add_special_tokens=False
                )["input_ids"]
                for ids in encoded:
                    token_buffer.extend(ids)
                while len(token_buffer) >= target_len:
                    batch_tokens = token_buffer[:target_len]
                    token_buffer = token_buffer[target_len:]
                    yield torch.tensor(batch_tokens, dtype=torch.long).reshape(
                        self.batch_size, self.seq_len
                    )

            epoch += 1

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Return next activation batch flattened to [N, d_model]."""
        if self._token_iter is None:
            self._token_iter = self._token_generator()

        tokens = next(self._token_iter).to(self.device)
        self.model(tokens)
        acts = self._captured_activations

        return acts.reshape(-1, acts.shape[-1]).float()

    def token_iterator(self) -> Iterator[torch.Tensor]:
        """Return a fresh token batch iterator (independent of the internal one used by next_batch)."""
        return self._token_generator()

    @property
    def last_activations(self) -> torch.Tensor | None:
        """Last captured residual-stream activations [B, S, d], or None before first forward."""
        return self._captured_activations

    @property
    def target_module(self) -> torch.nn.Module:
        """The hooked transformer module (for patching)."""
        return self._hf_target_module

    def swap_hook(self, new_hook_fn):
        """Temporarily replace the activation-capture hook with a custom one.

        Returns a handle for the new hook. The caller must call handle.remove()
        and then self.restore_hook() when done.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
        return self._hf_target_module.register_forward_hook(new_hook_fn)

    def restore_hook(self) -> None:
        """Re-register the default activation-capture hook."""
        self._register_hf_hook(self.model)

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get W_vocab: the unembedding matrix [d_model, V]."""
        lm_head = self.model.get_output_embeddings()
        return lm_head.weight.T.float()

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size
