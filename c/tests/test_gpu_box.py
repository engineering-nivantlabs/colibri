import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gpu_box', Path(__file__).resolve().parents[1] / 'tools/gpu_box.py')
gpu_box = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_box)


class GpuBoxTests(unittest.TestCase):
    def test_refuses_low_headroom(self):
        with self.assertRaises(ValueError):
            gpu_box.make_plan('model', 9 * gpu_box.GIB, 6, 12, 8, 2048, 32)

    def test_tier_preserves_reserve_and_exclusive_cache(self):
        plan = gpu_box.make_plan('model', 18 * gpu_box.GIB, 6, 12, 8, 2048, 32)
        self.assertLessEqual(float(plan['env']['RAM_GB']) * 1e9, 12 * gpu_box.GIB)
        self.assertEqual(plan['env']['COLI_VK_TIER_EXCLUSIVE'], '1')
        self.assertEqual(plan['env']['COLI_VK_TIER_RESERVE_GB'], '3')

    def test_cpu_baseline_disables_both_gpu_backends(self):
        env = gpu_box.make_plan('model', 18 * gpu_box.GIB, 6, 12, 8, 2048, 32, True)['env']
        self.assertEqual(env['COLI_VULKAN'], '0')
        self.assertEqual(env['COLI_CUDA'], '0')


if __name__ == '__main__':
    unittest.main()
