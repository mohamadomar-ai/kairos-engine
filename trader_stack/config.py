"""Central configuration. One place to change model sizes, devices, paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pulls .env into os.environ if present


def _detect_device() -> str:
    """Pick the best available torch device. Falls back to cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # --- LLM (the brain) ---------------------------------------------------
    # Gemma 4 via Ollama's OpenAI-compatible endpoint.
    # 26B MoE for deep reasoning, E4B for quick tasks. If 26B isn't available
    # (light hardware, pull failed), set deep_think_llm = "gemma4:e4b" too.
    llm_provider: str = "ollama"
    deep_think_llm: str = os.getenv("DEEP_THINK_LLM", "gemma4:26b")
    quick_think_llm: str = os.getenv("QUICK_THINK_LLM", "gemma4:e4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # --- Kronos ------------------------------------------------------------
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_model: str = "NeoQuasar/Kronos-small"  # 24.7M params, max_context=512
    kronos_max_context: int = 512
    kronos_lookback_bars: int = 400      # daily bars fed to Kronos
    kronos_pred_bars: int = 30           # ~6 weeks of daily forecast
    kronos_temperature: float = 1.0
    kronos_top_p: float = 0.9
    kronos_sample_count: int = 5         # average multiple paths for stability

    # --- TimesFM -----------------------------------------------------------
    timesfm_model: str = "google/timesfm-2.5-200m-pytorch"
    timesfm_max_context: int = 1024
    timesfm_max_horizon: int = 256
    timesfm_horizon: int = 30            # bars to forecast

    # --- Behaviour ---------------------------------------------------------
    force_forecast: bool = field(
        default_factory=lambda: os.getenv("FORCE_FORECAST", "false").lower() == "true"
    )
    max_debate_rounds: int = int(os.getenv("MAX_DEBATE_ROUNDS", "1"))

    # --- Filesystem --------------------------------------------------------
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("TRADINGAGENTS_CACHE_DIR", Path.home() / ".tradingagents" / "cache")
        )
    )

    # --- Compute -----------------------------------------------------------
    device: str = field(default_factory=_detect_device)

    def as_tradingagents_config(self) -> dict:
        """Build the dict that TradingAgents' DEFAULT_CONFIG.copy() wants."""
        # Imported here so import-time circulars don't bite us.
        try:
            from tradingagents.default_config import DEFAULT_CONFIG
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "TradingAgents is not installed. Run ./setup.sh first."
            ) from e

        cfg = DEFAULT_CONFIG.copy()
        cfg["llm_provider"] = self.llm_provider
        cfg["backend_url"] = self.ollama_base_url  # TradingAgents reads this for ollama
        cfg["deep_think_llm"] = self.deep_think_llm
        cfg["quick_think_llm"] = self.quick_think_llm
        cfg["max_debate_rounds"] = self.max_debate_rounds
        return cfg


CONFIG = Config()
