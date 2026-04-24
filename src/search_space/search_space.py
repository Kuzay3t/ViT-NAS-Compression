# src/search_space/search_space.py
import yaml
import random
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ArchitectureConfig:
    """Architecture configuration."""
    depth: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    patch_size: int
    dropout: float
    attn_dropout: float

@dataclass
class PruningConfig:
    """Pruning configuration."""
    enabled: bool
    method: str
    layer_wise_ratios: List[float]
    structured_granularity: str

@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    enabled: bool
    layer_wise_bits: List[int]
    quantize_weights: bool
    quantize_activations: bool
    symmetric: bool
    per_channel: bool

@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    enabled: bool
    apply_to_layers: str
    temperature: float
    kd_weight: float
    teacher_config: str

@dataclass
class CompressionConfig:
    """Complete compression configuration."""
    pruning: PruningConfig
    quantization: QuantizationConfig
    distillation: DistillationConfig

@dataclass
class AdaptivityConfig:
    """Adaptivity configuration."""
    enabled: bool
    mechanism: str
    early_exit: Dict[str, Any]
    token_dropping: Dict[str, Any]
    gating: Dict[str, Any]

@dataclass
class HardwareConfig:
    """Hardware constraints."""
    target_devices: List[str]
    target_latency_ms: float
    max_memory_mb: float
    max_model_size_mb: float
    max_energy_mj: float

@dataclass
class SearchConfig:
    """Complete search configuration."""
    architecture: ArchitectureConfig
    compression: CompressionConfig
    adaptivity: AdaptivityConfig
    hardware: HardwareConfig


class SearchSpace:
    """Defines and manages the unified search space for ViT NAS + Compression."""
    
    def __init__(self, config_path: Union[str, Path] = "config/search_space.yaml"):
        """Initialize search space from YAML configuration."""
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self.search_space_dict = self._build_search_space()
        logger.info(f"SearchSpace initialized from {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _build_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Build searchable parameters from raw config."""
        searchable = {}
        
        for key, spec in self.raw_config['architecture'].items():
            if 'choices' in spec:
                searchable[f'arch_{key}'] = spec['choices']
        
        searchable['comp_pruning_method'] = self.raw_config['compression']['pruning']['method']['choices']
        searchable['adapt_mechanism'] = self.raw_config['adaptivity']['mechanism']['choices']
        
        return searchable
    
    def random_sample(self) -> SearchConfig:
        """Sample a random configuration from the search space."""
        arch = ArchitectureConfig(
            depth=random.choice(self.raw_config['architecture']['depth']['choices']),
            embed_dim=random.choice(self.raw_config['architecture']['embed_dim']['choices']),
            num_heads=random.choice(self.raw_config['architecture']['num_heads']['choices']),
            mlp_ratio=random.choice(self.raw_config['architecture']['mlp_ratio']['choices']),
            patch_size=random.choice(self.raw_config['architecture']['patch_size']['choices']),
            dropout=random.uniform(0.0, 0.5),
            attn_dropout=random.uniform(0.0, 0.5),
        )
        
        num_layers = arch.depth
        
        pruning = PruningConfig(
            enabled=random.choice([True, False]),
            method=random.choice(self.raw_config['compression']['pruning']['method']['choices']),
            layer_wise_ratios=[random.uniform(0.0, 0.5) for _ in range(num_layers)],
            structured_granularity=random.choice(self.raw_config['compression']['pruning']['structured_granularity']['choices']),
        )
        
        quantization = QuantizationConfig(
            enabled=random.choice([True, False]),
            layer_wise_bits=[random.choice([4, 6, 8, 16]) for _ in range(num_layers)],
            quantize_weights=True,
            quantize_activations=True,
            symmetric=random.choice([True, False]),
            per_channel=True,
        )
        
        distillation = DistillationConfig(
            enabled=random.choice([True, False]),
            apply_to_layers=random.choice(self.raw_config['compression']['distillation']['apply_to_layers']['choices']),
            temperature=random.uniform(1.0, 20.0),
            kd_weight=random.uniform(0.0, 1.0),
            teacher_config=random.choice(self.raw_config['compression']['distillation']['teacher_config']['choices']),
        )
        
        compression = CompressionConfig(
            pruning=pruning,
            quantization=quantization,
            distillation=distillation,
        )
        
        adaptivity = AdaptivityConfig(
            enabled=random.choice([True, False]),
            mechanism=random.choice(self.raw_config['adaptivity']['mechanism']['choices']),
            early_exit={
                'enabled': random.choice([True, False]),
                'exit_points': random.sample(range(3, num_layers), k=random.randint(1, min(2, num_layers-3))) if num_layers > 3 else [],
                'threshold': random.uniform(0.1, 0.9),
            },
            token_dropping={
                'enabled': random.choice([True, False]),
                'drop_ratios': [random.uniform(0.0, 0.3) for _ in range(num_layers)],
            },
            gating={
                'enabled': random.choice([True, False]),
                'placement': random.choice(self.raw_config['adaptivity']['gating']['placement']['choices']),
                'threshold': random.uniform(0.0, 1.0),
            },
        )
        
        hardware = HardwareConfig(
            target_devices=random.sample(
                self.raw_config['hardware']['target_devices']['choices'],
                k=random.randint(1, 2)
            ),
            target_latency_ms=random.uniform(50.0, 200.0),
            max_memory_mb=random.choice([256, 512, 1024]),
            max_model_size_mb=random.choice([10, 25, 50, 100]),
            max_energy_mj=random.uniform(50.0, 200.0),
        )
        
        return SearchConfig(
            architecture=arch,
            compression=compression,
            adaptivity=adaptivity,
            hardware=hardware,
        )
    
    def to_dict(self, config: SearchConfig) -> Dict[str, Any]:
        """Convert SearchConfig to dictionary."""
        return {
            'architecture': asdict(config.architecture),
            'compression': {
                'pruning': asdict(config.compression.pruning),
                'quantization': asdict(config.compression.quantization),
                'distillation': asdict(config.compression.distillation),
            },
            'adaptivity': {
                'enabled': config.adaptivity.enabled,
                'mechanism': config.adaptivity.mechanism,
                'early_exit': config.adaptivity.early_exit,
                'token_dropping': config.adaptivity.token_dropping,
                'gating': config.adaptivity.gating,
            },
            'hardware': asdict(config.hardware),
        }
    
    def validate_config(self, config: SearchConfig) -> Tuple[bool, List[str]]:
        """Validate a configuration against constraints."""
        errors = []
        
        if config.architecture.embed_dim % config.architecture.num_heads != 0:
            errors.append(f"embed_dim must be divisible by num_heads")
        
        if any(r < 0 or r > 1 for r in config.compression.pruning.layer_wise_ratios):
            errors.append("Pruning ratios must be in [0, 1]")
        
        if any(b not in [4, 6, 8, 16] for b in config.compression.quantization.layer_wise_bits):
            errors.append("Quantization bit-widths must be in [4, 6, 8, 16]")
        
        return len(errors) == 0, errors
    
    def print_search_space_info(self):
        """Print search space statistics."""
        print("\n" + "="*70)
        print("SEARCH SPACE INFORMATION")
        print("="*70)
        
        print("\n--- ARCHITECTURE ---")
        arch_spec = self.raw_config['architecture']
        for param, spec in arch_spec.items():
            if 'choices' in spec:
                choices = spec['choices']
                print(f"{param:.<30} {len(choices)} choices: {choices}")
        
        print("\n--- COMPRESSION ---")
        print(f"{'Pruning methods':.<30} {len(self.raw_config['compression']['pruning']['method']['choices'])} methods")
        
        print("\n--- HARDWARE ---")
        print(f"{'Target devices':.<30} {len(self.raw_config['hardware']['target_devices']['choices'])} devices")
        
        print("="*70 + "\n")
