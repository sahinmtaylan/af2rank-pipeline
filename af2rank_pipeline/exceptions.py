class PipelineError(Exception):
    """Base exception for expected pipeline failures."""


class DependencyError(PipelineError):
    """Raised when an optional runtime dependency is unavailable."""


class CleaningError(PipelineError):
    """Raised when a raw model cannot be cleaned safely."""


class ChainMappingError(CleaningError):
    """Raised when raw chains cannot be mapped to the target spec."""
