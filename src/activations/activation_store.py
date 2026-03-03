"""Activation streaming via HuggingFace model hooks."""

from collections.abc import Iterator

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import DEVICE


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
        dataset_config: str | None = None,
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self._hf_target_module = dict(self.model.named_modules())[self.hook_point]

        def hook_fn(_module, _input, output):
            self._captured_activations = output[0].detach()

        self._hook_handle = self._hf_target_module.register_forward_hook(hook_fn)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._token_iter: Iterator[torch.Tensor] = self._token_generator()

    def _token_generator(self, batch_size: int | None = None) -> Iterator[torch.Tensor]:
        """Yield [batch_size, seq_len] token batches via HF streaming + batched tokenization."""
        bs = batch_size if batch_size is not None else self.batch_size
        target_len = bs * self.seq_len
        epoch = 0

        while True:
            dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=self.dataset_split,
                streaming=True,
            )
            dataset = dataset.select_columns([self.text_column])
            dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)
            dataset.set_epoch(epoch)

            tokenized = dataset.map(
                lambda ex: self.tokenizer(ex[self.text_column], add_special_tokens=False),
                batched=True,
                remove_columns=[self.text_column],
            )

            token_buffer: list[int] = []
            for example in tokenized:
                token_buffer.extend(example["input_ids"])
                while len(token_buffer) >= target_len:
                    batch_tokens = token_buffer[:target_len]
                    token_buffer = token_buffer[target_len:]
                    yield torch.tensor(batch_tokens, dtype=torch.long).reshape(
                        bs, self.seq_len
                    )

            epoch += 1

    @torch.no_grad()
    def next_batch(self) -> torch.Tensor:
        """Return next activation batch flattened to [N, d_model]."""
        tokens = next(self._token_iter).to(self.device)
        self.model(tokens)
        acts = self._captured_activations

        return acts.reshape(-1, acts.shape[-1]).float()

    def token_iterator(self, batch_size: int | None = None) -> Iterator[torch.Tensor]:
        """Return a fresh token batch iterator (independent of the internal one used by next_batch)."""
        return self._token_generator(batch_size)

    @property
    def last_activations(self) -> torch.Tensor:
        return self._captured_activations

    def swap_hook(self, new_hook_fn):
        """Replace the activation-capture hook. Caller must call handle.remove() + restore_hook()."""
        self._hook_handle.remove()
        return self._hf_target_module.register_forward_hook(new_hook_fn)

    def restore_hook(self) -> None:
        """Re-register the default activation-capture hook."""
        def hook_fn(_module, _input, output):
            self._captured_activations = output[0].detach()

        self._hook_handle = self._hf_target_module.register_forward_hook(hook_fn)

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get W_vocab: the unembedding matrix [d_model, V]."""
        lm_head = self.model.get_output_embeddings()
        return lm_head.weight.T.float()

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size
