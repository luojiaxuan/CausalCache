from __future__ import annotations

import math
import unittest

from causalcache.restoration_v2_gpu_kl import (
    CANDIDATE_LOGITS_DTYPE,
    COMPUTE_DTYPE,
    CPU_ORACLE_EQUIVALENCE_ATOL,
    CPU_ORACLE_EQUIVALENCE_RTOL,
    REFERENCE_DTYPE,
    cpu_full_vocab_mean_kl_oracle_for_tests,
    gpu_resident_full_vocab_mean_kl,
)


class RestorationV2CPUOracleTest(unittest.TestCase):
    def test_known_case_and_log_prob_candidate_agree(self) -> None:
        reference = [
            [
                [math.log(0.75), math.log(0.25)],
                [math.log(0.2), math.log(0.8)],
            ]
        ]
        candidate_logits = [[[0.0, 0.0], [math.log(0.4), math.log(0.6)]]]
        candidate_log_probs = [
            [
                [math.log(0.5), math.log(0.5)],
                [math.log(0.4), math.log(0.6)],
            ]
        ]
        from_logits = cpu_full_vocab_mean_kl_oracle_for_tests(
            reference,
            candidate_logits,
            candidate_representation="logits",
        )
        from_log_probs = cpu_full_vocab_mean_kl_oracle_for_tests(
            reference,
            candidate_log_probs,
            candidate_representation="log_probs",
        )
        expected_first = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
        expected_second = 0.2 * math.log(0.5) + 0.8 * math.log(4.0 / 3.0)
        self.assertAlmostEqual(from_logits[0], (expected_first + expected_second) / 2)
        self.assertEqual(from_logits, from_log_probs)

    def test_oracle_rejects_shape_nonfinite_normalization_and_kind_drift(self) -> None:
        valid = [[[math.log(0.5), math.log(0.5)]]]
        cases = (
            ([], valid, "logits"),
            (valid, [[[0.0]]], "logits"),
            (valid, [[[0.0, math.inf]]], "logits"),
            (valid, [[[0.0, 0.0]]], "log_probs"),
        )
        for reference, candidate, representation in cases:
            with self.subTest(
                reference=reference,
                candidate=candidate,
                representation=representation,
            ):
                with self.assertRaises((TypeError, ValueError)):
                    cpu_full_vocab_mean_kl_oracle_for_tests(
                        reference,
                        candidate,
                        candidate_representation=representation,
                    )
        with self.assertRaisesRegex(ValueError, "candidate_representation"):
            cpu_full_vocab_mean_kl_oracle_for_tests(
                valid,
                valid,
                candidate_representation="probabilities",  # type: ignore[arg-type]
            )


class RestorationV2GPUKLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("PyTorch is an optional GPU runtime dependency") from error
        cls.torch = torch

    def test_canonical_path_rejects_cpu_and_dtype_drift_before_computation(self) -> None:
        torch = self.torch
        reference = torch.log_softmax(torch.tensor([[[1.0, 2.0]]]), dim=-1)
        candidate = torch.tensor([[[1.0, 2.0]]], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA-resident"):
            gpu_resident_full_vocab_mean_kl(
                reference,
                candidate,
                candidate_representation="logits",
            )
        with self.assertRaisesRegex(TypeError, "reference_log_probs dtype drifted"):
            gpu_resident_full_vocab_mean_kl(
                reference.to(dtype=torch.float64),
                candidate,
                candidate_representation="logits",
            )
        with self.assertRaisesRegex(TypeError, "candidate dtype drifted"):
            gpu_resident_full_vocab_mean_kl(
                reference,
                candidate.float(),
                candidate_representation="logits",
            )
        with self.assertRaisesRegex(ValueError, "rank-3"):
            gpu_resident_full_vocab_mean_kl(
                reference[0],
                candidate[0],
                candidate_representation="logits",
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            gpu_resident_full_vocab_mean_kl(
                reference,
                candidate.repeat(2, 1, 1),
                candidate_representation="logits",
            )

    def test_cuda_known_case_device_output_audit_and_validation(self) -> None:
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the canonical GPU path")
        device = torch.device("cuda:0")
        reference_logits = torch.tensor(
            [
                [[1.0, 0.0, -1.0], [0.25, -0.5, 1.5]],
                [[-0.5, 0.5, 1.0], [2.0, 0.0, -0.25]],
            ],
            dtype=torch.float32,
            device=device,
        )
        reference = torch.log_softmax(reference_logits, dim=-1)
        candidate_logits = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, -0.25, 0.75]],
                [[0.0, 1.0, -0.5], [1.0, 0.5, -1.0]],
            ],
            dtype=torch.bfloat16,
            device=device,
        )
        result = gpu_resident_full_vocab_mean_kl(
            reference,
            candidate_logits,
            candidate_representation="logits",
        )
        oracle = cpu_full_vocab_mean_kl_oracle_for_tests(
            reference.detach().cpu().tolist(),
            candidate_logits.detach().float().cpu().tolist(),
            candidate_representation="logits",
        )
        self.assertEqual(tuple(result.per_example_mean_kl.shape), (2,))
        self.assertEqual(result.per_example_mean_kl.device, device)
        self.assertEqual(str(result.per_example_mean_kl.dtype), COMPUTE_DTYPE)
        for actual, expected in zip(
            result.per_example_mean_kl.detach().cpu().tolist(),
            oracle,
            strict=True,
        ):
            self.assertTrue(
                math.isclose(
                    actual,
                    expected,
                    abs_tol=CPU_ORACLE_EQUIVALENCE_ATOL,
                    rel_tol=CPU_ORACLE_EQUIVALENCE_RTOL,
                )
            )
        audit = result.audit.to_dict()
        self.assertEqual(audit["reference_input_dtype"], REFERENCE_DTYPE)
        self.assertEqual(audit["candidate_input_dtype"], CANDIDATE_LOGITS_DTYPE)
        self.assertEqual(audit["compute_dtype"], COMPUTE_DTYPE)
        self.assertEqual(audit["validation_scalar_host_reads"], 0)
        self.assertEqual(audit["full_tensor_host_transfers"], 0)
        self.assertEqual(audit["numeric_validation"], "gpu_resident_per_example_predicates")
        self.assertEqual(audit["invalid_numeric_output"], "nan_final_distance")
        self.assertEqual(audit["device_validation_category_count"], 4)
        self.assertFalse(audit["reference_zero_copy_batch_expansion"])
        self.assertEqual(audit["reference_compute_batch_size"], 2)

        shared_reference = reference[:1].expand(2, -1, -1)
        self.assertEqual(shared_reference.stride(0), 0)
        self.assertEqual(
            shared_reference.untyped_storage().data_ptr(),
            reference.untyped_storage().data_ptr(),
        )
        shared_result = gpu_resident_full_vocab_mean_kl(
            shared_reference,
            candidate_logits,
            candidate_representation="logits",
        )
        self.assertEqual(tuple(shared_result.per_example_mean_kl.shape), (2,))
        self.assertEqual(shared_result.audit.reference_batch_stride, 0)
        self.assertTrue(shared_result.audit.reference_zero_copy_batch_expansion)
        self.assertEqual(shared_result.audit.reference_compute_batch_size, 1)

        candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
        log_prob_result = gpu_resident_full_vocab_mean_kl(
            reference,
            candidate_log_probs,
            candidate_representation="log_probs",
        )
        self.assertTrue(
            torch.allclose(
                result.per_example_mean_kl,
                log_prob_result.per_example_mean_kl,
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertEqual(
            log_prob_result.audit.candidate_input_dtype,
            COMPUTE_DTYPE,
        )

        invalid_nonfinite = gpu_resident_full_vocab_mean_kl(
            reference,
            candidate_logits.clone().index_fill_(
                2,
                torch.tensor([0], device=device),
                math.inf,
            ),
            candidate_representation="logits",
        )
        self.assertFalse(torch.isfinite(invalid_nonfinite.per_example_mean_kl).all())
        with self.assertRaisesRegex(TypeError, "candidate dtype drifted"):
            gpu_resident_full_vocab_mean_kl(
                reference,
                candidate_logits.float(),
                candidate_representation="logits",
            )
        invalid_normalization = gpu_resident_full_vocab_mean_kl(
            torch.zeros_like(reference),
            candidate_logits,
            candidate_representation="logits",
        )
        self.assertFalse(
            torch.isfinite(invalid_normalization.per_example_mean_kl).all()
        )
        if torch.cuda.device_count() >= 2:
            with self.assertRaisesRegex(ValueError, "same CUDA device"):
                gpu_resident_full_vocab_mean_kl(
                    reference,
                    candidate_logits.to("cuda:1"),
                    candidate_representation="logits",
                )


if __name__ == "__main__":
    unittest.main()
