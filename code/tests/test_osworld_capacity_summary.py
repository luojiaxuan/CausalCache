from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.summarize_osworld_capacity_sweep import _cpu_telemetry, _gpu_telemetry


class OSWorldCapacitySummaryTests(unittest.TestCase):
    def test_filters_gpu_samples_to_active_replicas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia-smi.csv"
            path.write_text(
                "2026/07/22 00:00:00, 0, 10, 18000, 100\n"
                "2026/07/22 00:00:00, 1, 99, 19000, 200\n"
                "2026/07/22 00:00:01, 0, 30, 18000, 120\n",
                encoding="utf-8",
            )
            value = _gpu_telemetry(path, replicas=1)
        self.assertEqual(value["sample_count"], 2)
        self.assertEqual(value["utilization_mean_percent"], 20)
        self.assertEqual(value["memory_max_mib"], 18000)

    def test_omits_vmstat_boot_average(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vmstat.txt"
            path.write_text(
                "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n"
                " r  b swpd free buff cache si so bi bo in cs us sy id wa st\n"
                "1 0 0 1 1 1 0 0 0 0 0 0 90 0 10 0 0\n"
                "3 0 0 1 1 1 0 0 0 0 0 0 20 5 74 1 0\n",
                encoding="utf-8",
            )
            value = _cpu_telemetry(path)
        self.assertEqual(value["sample_count"], 1)
        self.assertEqual(value["runnable_mean"], 3)
        self.assertEqual(value["cpu_idle_mean_percent"], 74)


if __name__ == "__main__":
    unittest.main()
