from .expression_adapter import ExpressionFiLMAdapter, ExpressionResidualAdapter
from .expression_control import (
    DEFAULT_CONTROL,
    apply_expression_controls,
    format_expression_stats,
    load_expression_control,
    summarize_expression_features,
)
from .expression_encoder import ExpressionEncoder, ExpressionFeaturePredictor
from .expression_feature import extract_expression_features
from .differentiable_features import (
    differentiable_mel_axes,
    differentiable_onset_effort_features,
)
from .intensity_adapter import ScalarIntensityAdapter, SymmetricIntensityAdapter
from .vocal_effort_adapter import SymmetricVocalEffortAdapter
from .vocal_effort_feature import (
    EFFORT_FEATURE_NAMES,
    apply_vocal_effort_transform,
    mel_active_mask,
    mel_effort_features,
    vocal_effort_feature_loss,
)
from .hierarchical_adapter import (
    HierarchicalExpressionAdapter,
    axis_direction_orthogonality_loss,
)
from .hierarchical_features import (
    AXIS_NAMES,
    COMPONENT_NAMES,
    compose_axis_targets,
    extract_hierarchical_components,
    extract_structure_gates,
)
from .hierarchical_representation import HierarchicalRepresentationEncoder
from .onset_accent_controller import (
    apply_onset_accent,
    load_onset_times,
    onset_activity,
)
from .asymmetric_onset_effort_adapter import (
    AsymmetricOnsetEffortAdapter,
    AsymmetricOnsetExpressionAdapter,
)
from .range_conditioned_onset_adapter import (
    RangeConditionedOnsetAdapter,
    RangeConditionedExpressionAdapter,
    pitch_range_features,
)

__all__ = [
    "DEFAULT_CONTROL",
    "ExpressionFiLMAdapter",
    "ExpressionResidualAdapter",
    "ScalarIntensityAdapter",
    "SymmetricIntensityAdapter",
    "SymmetricVocalEffortAdapter",
    "ExpressionEncoder",
    "ExpressionFeaturePredictor",
    "apply_expression_controls",
    "extract_expression_features",
    "differentiable_mel_axes",
    "differentiable_onset_effort_features",
    "format_expression_stats",
    "load_expression_control",
    "summarize_expression_features",
    "EFFORT_FEATURE_NAMES",
    "apply_vocal_effort_transform",
    "mel_active_mask",
    "mel_effort_features",
    "vocal_effort_feature_loss",
    "AXIS_NAMES",
    "COMPONENT_NAMES",
    "HierarchicalExpressionAdapter",
    "HierarchicalRepresentationEncoder",
    "axis_direction_orthogonality_loss",
    "compose_axis_targets",
    "extract_hierarchical_components",
    "extract_structure_gates",
    "apply_onset_accent",
    "load_onset_times",
    "onset_activity",
    "AsymmetricOnsetEffortAdapter",
    "AsymmetricOnsetExpressionAdapter",
    "RangeConditionedOnsetAdapter",
    "RangeConditionedExpressionAdapter",
    "pitch_range_features",
]
