import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location('anima_cache', Path(__file__).parents[1] / 'ExtraNodes/SwarmAnimaNodes/cache.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = module.ArtifactCache(self.root / 'cache')

    def entry(self, key):
        return self.cache.get('pose', key, lambda: {'value': torch.tensor([int(key)])})

    def test_hit_skips_compute_and_survives_restart(self):
        expected = self.entry('1')
        restarted = module.ArtifactCache(self.root / 'cache')
        actual = restarted.get('pose', '1', lambda: self.fail('Cache hit recomputed'))
        self.assertTrue(torch.equal(expected['value'], actual['value']))

    def test_lru_evicts_oldest_not_last_read(self):
        self.cache.policy['max_entries'] = 2
        self.entry('1')
        self.entry('2')
        os.utime(self.root / 'cache/pose-2.safetensors', (1, 1))
        self.entry('1')
        self.entry('3')
        self.assertTrue((self.root / 'cache/pose-1.safetensors').exists())
        self.assertFalse((self.root / 'cache/pose-2.safetensors').exists())

    def test_disabled_cache_never_writes_tensor(self):
        self.cache.policy['max_mb'] = 0
        self.entry('1')
        self.assertEqual(list((self.root / 'cache').glob('*.safetensors')), [])

    def test_oversized_entry_is_evicted(self):
        self.cache.policy['max_mb'] = 0.00001
        self.entry('1')
        self.assertEqual(list((self.root / 'cache').glob('*.safetensors')), [])

    def test_stale_and_corrupt_entries_recompute(self):
        self.entry('1')
        path = self.root / 'cache/pose-1.safetensors'
        path.write_bytes(b'broken')
        self.assertEqual(self.entry('1')['value'].item(), 1)
        self.cache.policy['max_age_days'] = 1
        os.utime(path, (1, 1))
        self.cache.get('pose', '1', lambda: {'value': torch.tensor([42])})
        self.assertEqual(self.entry('1')['value'].item(), 42)

    def test_key_changes_with_image_version_and_model(self):
        image = torch.zeros(1, 16, 16, 3)
        weights = self.root / 'model'
        weights.write_bytes(b'a')
        key = module.image_key(image, 'v1', [weights])
        self.assertEqual(key, module.image_key(image.clone(), 'v1', [weights]))
        self.assertNotEqual(key, module.image_key(image + 1, 'v1', [weights]))
        self.assertNotEqual(key, module.image_key(image, 'v2', [weights]))
        weights.write_bytes(b'changed')
        self.assertNotEqual(key, module.image_key(image, 'v1', [weights]))


if __name__ == '__main__':
    unittest.main()
