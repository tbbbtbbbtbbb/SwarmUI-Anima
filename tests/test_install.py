from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

spec = importlib.util.spec_from_file_location('anima_install', Path(__file__).parents[1] / 'ExtraNodes/SwarmAnimaNodes/install.py')
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


class InstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models = self.root / 'models'
        self.target = self.models / 'anima/fixture.bin'
        self.data = b'verified model bytes'
        manifest = [dict(path='anima/fixture.bin', repo='test/model', revision='pinned', source='weights.bin',
                         size=len(self.data), sha256=hashlib.sha256(self.data).hexdigest())]
        (self.root / 'models.json').write_text(json.dumps(manifest))
        source = patch.object(install, '__file__', str(self.root / 'install.py'))
        source.start()
        self.addCleanup(source.stop)

    def test_download_and_offline_restart(self):
        with patch.object(install.urllib.request, 'urlopen', return_value=io.BytesIO(self.data)) as download:
            install.ensure_models(self.models)
            download.assert_called_once_with('https://huggingface.co/test/model/resolve/pinned/weights.bin', timeout=180)
        self.assertEqual(self.target.read_bytes(), self.data)
        with patch.object(install.urllib.request, 'urlopen', side_effect=AssertionError('Restart attempted network')):
            install.ensure_models(self.models)

    def test_existing_different_file_is_preserved(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'user model')
        with patch.object(install.urllib.request, 'urlopen', side_effect=AssertionError('Unexpected download')):
            with self.assertRaisesRegex(RuntimeError, 'not overwritten'):
                install.ensure_models(self.models)
        self.assertEqual(self.target.read_bytes(), b'user model')

    def test_corrupt_download_is_removed_and_retry_succeeds(self):
        with patch.object(install.urllib.request, 'urlopen', return_value=io.BytesIO(b'x' * len(self.data))):
            with self.assertRaisesRegex(RuntimeError, 'corrupt'):
                install.ensure_models(self.models)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix('.bin.download').exists())
        with patch.object(install.urllib.request, 'urlopen', return_value=io.BytesIO(self.data)):
            install.ensure_models(self.models)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_network_failure_does_not_leave_partial_model(self):
        with patch.object(install.urllib.request, 'urlopen', side_effect=URLError('offline')):
            with self.assertRaises(URLError):
                install.ensure_models(self.models)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix('.bin.download').exists())

    def test_concurrent_backends_download_once(self):
        with patch.object(install.urllib.request, 'urlopen', return_value=io.BytesIO(self.data)) as download:
            with ThreadPoolExecutor(2) as threads:
                list(threads.map(install.ensure_models, [self.models, self.models]))
            self.assertEqual(download.call_count, 1)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_missing_dependency_uses_backend_python_without_dependency_changes(self):
        with patch.object(install, 'version', side_effect=install.PackageNotFoundError), patch.object(install.subprocess, 'run') as pip:
            install.ensure_dependencies()
        command = pip.call_args.args[0]
        self.assertEqual(command[:4], [install.sys.executable, '-m', 'pip', 'install'])
        self.assertIn('--no-deps', command)
        self.assertEqual(command[-1], 'rtmlib==0.0.16')

    def test_existing_newer_dependency_is_left_alone(self):
        with patch.object(install, 'version', return_value='0.1.0'), patch.object(install.subprocess, 'run', side_effect=AssertionError('Unexpected pip')):
            install.ensure_dependencies()

    def test_external_dependencies_constrain_existing_packages(self):
        def check(command, **kwargs):
            self.assertIn('--only-binary=:all:', command)
            self.assertEqual(command[-2:], ['opencv-python-headless', 'onnxruntime'])
            constraints = Path(command[command.index('--constraint') + 1]).read_text()
            self.assertIn('torch==', constraints.lower())
        with patch.object(install, 'find_spec', return_value=None), patch.object(install, 'version', return_value='0.0.16'), patch.object(install.subprocess, 'run', side_effect=check):
            install.ensure_dependencies()


if __name__ == '__main__':
    unittest.main()
