"""GPU-resident full-vocabulary KL for restoration-v2 policy screening."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence


CandidateRepresentation = Literal["logits", "log_probs"]

REFERENCE_DTYPE = "torch.float32"
CANDIDATE_LOGITS_DTYPE = "torch.bfloat16"
CANDIDATE_LOG_PROBS_DTYPE = "torch.float32"
COMPUTE_DTYPE = "torch.float32"
LOG_NORMALIZATION_ATOL = 5e-4
NEGATIVE_KL_ATOL = 1e-5
CPU_ORACLE_EQUIVALENCE_ATOL = 1e-6
CPU_ORACLE_EQUIVALENCE_RTOL = 1e-5


@dataclass(frozen=True)
class FullVocabularyMeanKLAudit:
    """JSON-safe metadata for one canonical GPU KL call."""

    operation: str
    candidate_representation: CandidateRepresentation
    batch_size: int
    distance_tokens: int
    vocabulary_size: int
    device: str
    reference_input_dtype: str
    candidate_input_dtype: str
    compute_dtype: str
    output_dtype: str
    reference_batch_stride: int
    reference_zero_copy_batch_expansion: bool
    reference_compute_batch_size: int
    reduction: str
    log_normalization_atol: float
    negative_kl_atol: float
    maximum_log_normalization_error: float
    minimum_per_token_kl_before_clamp: float
    validation_scalar_host_reads: int
    full_tensor_host_transfers: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FullVocabularyMeanKLResult:
    """Per-example KL device scalars and their immutable audit record."""

    per_example_mean_kl: Any
    audit: FullVocabularyMeanKLAudit


def _torch_module() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("GPU restoration KL requires PyTorch") from error
    return torch


def _validated_candidate_representation(value: str) -> CandidateRepresentation:
    if value not in {"logits", "log_probs"}:
        raise ValueError("candidate_representation must be 'logits' or 'log_probs'")
    return value


def _validate_gpu_inputs(
    *,
    torch: Any,
    reference_log_probs: Any,
    candidate: Any,
    candidate_representation: CandidateRepresentation,
) -> tuple[int, int, int]:
    if not isinstance(reference_log_probs, torch.Tensor) or not isinstance(
        candidate, torch.Tensor
    ):
        raise TypeError("reference_log_probs and candidate must be PyTorch tensors")
    if reference_log_probs.ndim != 3 or candidate.ndim != 3:
        raise ValueError("reference_log_probs and candidate must have rank-3 [batch,tokens,vocab]")
    if reference_log_probs.shape != candidate.shape:
        raise ValueError("reference_log_probs and candidate must have the same shape")
    if any(int(size) <= 0 for size in reference_log_probs.shape):
        raise ValueError("batch, token, and vocabulary dimensions must be non-empty")
    if str(reference_log_probs.dtype) != REFERENCE_DTYPE:
        raise TypeError(f"reference_log_probs dtype drifted; expected {REFERENCE_DTYPE}")
    expected_candidate_dtype = (
        CANDIDATE_LOGITS_DTYPE
        if candidate_representation == "logits"
        else CANDIDATE_LOG_PROBS_DTYPE
    )
    if str(candidate.dtype) != expected_candidate_dtype:
        raise TypeError(
            "candidate dtype drifted for "
            f"{candidate_representation}; expected {expected_candidate_dtype}"
        )
    if reference_log_probs.requires_grad or candidate.requires_grad:
        raise ValueError("canonical restoration KL inputs must not require gradients")
    if reference_log_probs.device.type != "cuda" or candidate.device.type != "cuda":
        raise ValueError("canonical restoration KL requires CUDA-resident inputs")
    if reference_log_probs.device != candidate.device:
        raise ValueError("reference_log_probs and candidate must use the same CUDA device")
    batch_size, distance_tokens, vocabulary_size = (
        int(size) for size in reference_log_probs.shape
    )
    return batch_size, distance_tokens, vocabulary_size


def gpu_resident_full_vocab_mean_kl(
    reference_log_probs: Any,
    candidate: Any,
    *,
    candidate_representation: CandidateRepresentation,
) -> FullVocabularyMeanKLResult:
    """Compute mean teacher-forced full-vocabulary KL without full-tensor host copies.

    ``reference_log_probs`` must be normalized FP32 log-probabilities. Candidate
    logits must be BF16; candidate log-probabilities must already be normalized
    FP32. Both inputs have shape ``[batch, distance_tokens, vocabulary]`` on one
    explicit CUDA device. The returned tensor has shape ``[batch]``, remains on
    that device, and is FP32.

    A cached ``[1, tokens, vocab]`` reference may be reused for a candidate
    microbatch with ``reference.expand(batch, -1, -1)``. The resulting zero-stride
    view is accepted without a reference ``repeat`` or ``contiguous`` copy and is
    recorded in the audit metadata.

    Validation performs four explicit scalar device-to-host reads. No full input,
    intermediate, per-token, or output tensor is copied to CPU by this function.
    """

    torch = _torch_module()
    representation = _validated_candidate_representation(candidate_representation)
    batch_size, distance_tokens, vocabulary_size = _validate_gpu_inputs(
        torch=torch,
        reference_log_probs=reference_log_probs,
        candidate=candidate,
        candidate_representation=representation,
    )

    zero_copy_batch_expansion = (
        batch_size > 1 and int(reference_log_probs.stride(0)) == 0
    )
    # note (luojiaxuan): Collapse a zero-stride expanded reference before exp()
    # so its probability tensor is materialized once, then broadcast against the
    # candidate batch. The expanded input remains useful for an exact-shape API
    # without paying a B-fold reference-probability allocation.
    reference_for_compute = (
        reference_log_probs[:1]
        if zero_copy_batch_expansion
        else reference_log_probs
    )
    input_finite = torch.isfinite(reference_for_compute).all() & torch.isfinite(
        candidate
    ).all()
    if not bool(input_finite.item()):
        raise ValueError("restoration KL inputs contain non-finite values")

    with torch.inference_mode():
        if representation == "logits":
            candidate_log_probs = torch.log_softmax(
                candidate.to(dtype=torch.float32),
                dim=-1,
            )
        else:
            candidate_log_probs = candidate

        reference_normalization_error = torch.abs(
            torch.logsumexp(reference_for_compute, dim=-1)
        ).amax()
        candidate_normalization_error = torch.abs(
            torch.logsumexp(candidate_log_probs, dim=-1)
        ).amax()
        maximum_normalization_error = float(
            torch.maximum(
                reference_normalization_error,
                candidate_normalization_error,
            ).item()
        )
        if maximum_normalization_error > LOG_NORMALIZATION_ATOL:
            raise ValueError(
                "restoration KL log-probabilities are not normalized within "
                f"atol={LOG_NORMALIZATION_ATOL}"
            )

        per_token_kl = torch.sum(
            reference_for_compute.exp()
            * (reference_for_compute - candidate_log_probs),
            dim=-1,
            dtype=torch.float32,
        )
        minimum_per_token_kl = float(per_token_kl.amin().item())
        if minimum_per_token_kl < -NEGATIVE_KL_ATOL:
            raise ValueError(
                "restoration KL is materially negative; inputs may not be log-probabilities"
            )
        per_example_mean_kl = per_token_kl.clamp_min(0.0).mean(
            dim=-1,
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(per_example_mean_kl).all().item()):
            raise ValueError("restoration KL output contains non-finite values")

    if (
        per_example_mean_kl.shape != (batch_size,)
        or per_example_mean_kl.device != reference_log_probs.device
        or str(per_example_mean_kl.dtype) != COMPUTE_DTYPE
    ):
        raise RuntimeError("restoration KL output shape, device, or dtype drifted")

    audit = FullVocabularyMeanKLAudit(
        operation="teacher_forced_full_vocabulary_mean_kl_on_distance_token_span",
        candidate_representation=representation,
        batch_size=batch_size,
        distance_tokens=distance_tokens,
        vocabulary_size=vocabulary_size,
        device=str(reference_log_probs.device),
        reference_input_dtype=str(reference_log_probs.dtype),
        candidate_input_dtype=str(candidate.dtype),
        compute_dtype=COMPUTE_DTYPE,
        output_dtype=str(per_example_mean_kl.dtype),
        reference_batch_stride=int(reference_log_probs.stride(0)),
        reference_zero_copy_batch_expansion=zero_copy_batch_expansion,
        reference_compute_batch_size=int(reference_for_compute.shape[0]),
        reduction="full_vocabulary_sum_then_distance_token_mean_per_example",
        log_normalization_atol=LOG_NORMALIZATION_ATOL,
        negative_kl_atol=NEGATIVE_KL_ATOL,
        maximum_log_normalization_error=maximum_normalization_error,
        minimum_per_token_kl_before_clamp=minimum_per_token_kl,
        validation_scalar_host_reads=4,
        full_tensor_host_transfers=0,
    )
    return FullVocabularyMeanKLResult(
        per_example_mean_kl=per_example_mean_kl,
        audit=audit,
    )


def _validated_cpu_cube(
    values: Sequence[Sequence[Sequence[float]]],
    *,
    name: str,
) -> list[list[list[float]]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty rank-3 numeric sequence")
    cube: list[list[list[float]]] = []
    expected_tokens: int | None = None
    expected_vocabulary: int | None = None
    for example in values:
        if isinstance(example, (str, bytes)) or not isinstance(example, Sequence) or not example:
            raise ValueError(f"{name} must be a non-empty rectangular rank-3 sequence")
        if expected_tokens is None:
            expected_tokens = len(example)
        elif len(example) != expected_tokens:
            raise ValueError(f"{name} must be a non-empty rectangular rank-3 sequence")
        converted_example: list[list[float]] = []
        for row in example:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
                raise ValueError(f"{name} must be a non-empty rectangular rank-3 sequence")
            if expected_vocabulary is None:
                expected_vocabulary = len(row)
            elif len(row) != expected_vocabulary:
                raise ValueError(f"{name} must be a non-empty rectangular rank-3 sequence")
            converted_row: list[float] = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must contain only real numeric values")
                converted = float(value)
                if not math.isfinite(converted):
                    raise ValueError(f"{name} contains non-finite values")
                converted_row.append(converted)
            converted_example.append(converted_row)
        cube.append(converted_example)
    return cube


def _log_softmax_row(row: Sequence[float]) -> list[float]:
    maximum = max(row)
    log_partition = maximum + math.log(sum(math.exp(value - maximum) for value in row))
    return [value - log_partition for value in row]


def cpu_full_vocab_mean_kl_oracle_for_tests(
    reference_log_probs: Sequence[Sequence[Sequence[float]]],
    candidate: Sequence[Sequence[Sequence[float]]],
    *,
    candidate_representation: CandidateRepresentation,
) -> tuple[float, ...]:
    """Independent float64 CPU oracle; production screening must not call it."""

    representation = _validated_candidate_representation(candidate_representation)
    reference_cube = _validated_cpu_cube(
        reference_log_probs,
        name="reference_log_probs",
    )
    candidate_cube = _validated_cpu_cube(candidate, name="candidate")
    reference_shape = (
        len(reference_cube),
        len(reference_cube[0]),
        len(reference_cube[0][0]),
    )
    candidate_shape = (
        len(candidate_cube),
        len(candidate_cube[0]),
        len(candidate_cube[0][0]),
    )
    if reference_shape != candidate_shape:
        raise ValueError("reference_log_probs and candidate must have the same shape")

    means: list[float] = []
    for reference_example, candidate_example in zip(
        reference_cube,
        candidate_cube,
        strict=True,
    ):
        per_token: list[float] = []
        for reference_row, candidate_row in zip(
            reference_example,
            candidate_example,
            strict=True,
        ):
            candidate_log_probs = (
                _log_softmax_row(candidate_row)
                if representation == "logits"
                else candidate_row
            )
            maximum_normalization_error = max(
                abs(_log_softmax_row(reference_row)[0] - reference_row[0]),
                abs(_log_softmax_row(candidate_log_probs)[0] - candidate_log_probs[0]),
            )
            if maximum_normalization_error > LOG_NORMALIZATION_ATOL:
                raise ValueError(
                    "CPU oracle log-probabilities are not normalized within "
                    f"atol={LOG_NORMALIZATION_ATOL}"
                )
            value = sum(
                math.exp(reference_value)
                * (reference_value - candidate_value)
                for reference_value, candidate_value in zip(
                    reference_row,
                    candidate_log_probs,
                    strict=True,
                )
            )
            if value < -NEGATIVE_KL_ATOL:
                raise ValueError("CPU oracle KL is materially negative")
            per_token.append(max(0.0, value))
        means.append(sum(per_token) / len(per_token))
    return tuple(means)
