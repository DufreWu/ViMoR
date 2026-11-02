import gc
import torch 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_logger, torch_dtype_from_str

logger = get_logger(__name__)

class TokenPoolingFeatureExtractor:
    """
    Entropy-Guided Token Pooling feature extractor.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (Hub ID or local path).
    prompt_prefix_k : int
        Number of prefix tokens to use for pooling. The tokeniser will
        keep only the first *k* tokens (plus special tokens). If k == 0
        the BOS embedding is used as the sole representation.
    batch_size : int
        Tokenisation and inference batch size.
    device : torch.device | str
        Target device. ``"auto"`` selects CUDA when available, else CPU.
    torch_dtype : str | torch.dtype
        Weight precision for the backbone. Default: ``"bfloat16"``.
    temperature : float
        Temperature applied to entropy scores before softmax. Default: 1.0.
    trust_remote_code : bool
        Passed to HuggingFace ``from_pretrained``. Default: ``True``.
    """

    def __init__(
        self,
        model_id: str,
        prompt_prefix_k: int = 4,
        batch_size: int = 64,
        device: torch.device | str = "auto",
        torch_dtype: str | torch.dtype = "bfloat16",
        temperature: float = 1.0,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.prompt_prefix_k = max(prompt_prefix_k, 1)  # at least 1 token
        self.batch_size = batch_size
        self.temperature = temperature

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve dtype
        if isinstance(torch_dtype, str):
            torch_dtype = torch_dtype_from_str(torch_dtype)
        self.torch_dtype = torch_dtype

        self.trust_remote_code = trust_remote_code
        self._backbone: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Backbone lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def load_backbone(self) -> None:
        """Load the tokeniser and backbone model into memory."""
        logger.info(f"Loading backbone: {self.model_id!r} (dtype={self.torch_dtype})")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._backbone = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self._backbone.eval()
        logger.info("Backbone loaded successfully.")

    def unload_backbone(self) -> None:
        """Delete the backbone from GPU memory and collect garbage."""
        if self._backbone is not None:
            del self._backbone
            self._backbone = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Backbone unloaded; GPU memory released.")

    # ──────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Run one batch through the backbone and return entropy-pooled features.

        Parameters
        ----------
        texts : list[str]  – batch of prompt strings.

        Returns
        -------
        pooled : torch.Tensor of shape (B, d)  on CPU, float32.
        """
        assert self._backbone is not None and self._tokenizer is not None, (
            "Call load_backbone() before extract()."
        )

        enc = self._tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.prompt_prefix_k,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # ── last hidden states ── shape (B, k, d)
        last_hs = outputs.hidden_states[-1]

        # ── per-token Shannon entropy ── shape (B, k)
        probs = F.softmax(outputs.logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # ── entropy-based attention weights ──
        scaled_e = entropy / self.temperature
        scaled_e = scaled_e.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(scaled_e, dim=1)  # (B, k)

        # ── weighted pooling ── shape (B, d)
        pooled = torch.sum(last_hs * weights.unsqueeze(-1), dim=1)

        return pooled.cpu().float()

    def extract(self, prompts: list[str]) -> torch.Tensor:
        """
        Extract features for a list of prompt strings.

        Automatically loads the backbone if not already loaded.

        Parameters
        ----------
        prompts : list[str]
            Raw prompt texts. These should be the ``user_prompt_content``
            column from the ForeLen CSVs.

        Returns
        -------
        features : torch.Tensor of shape (N, d)  – pooled representations,
                   on CPU, dtype float32.
        """
        if self._backbone is None:
            self.load_backbone()

        all_features: List[torch.Tensor] = []
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            feats = self._extract_batch(batch)
            all_features.append(feats)

            batch_idx = i // self.batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                logger.info(f"  Extracted {min(i + self.batch_size, len(prompts))}/{len(prompts)} samples")

        return torch.cat(all_features, dim=0)  # (N, d)