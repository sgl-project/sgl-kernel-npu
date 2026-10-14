"""CPU-only guards for the isolated A2 regression loader."""
import ctypes
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fuseep_a2_loader import load_private_opapi


class LoaderTests(unittest.TestCase):
    def test_rejects_relative_or_missing_path(self):
        for path in ('relative.so', '/nonexistent/fuseep-test.so'):
            with self.assertRaises(RuntimeError):
                load_private_opapi(path)

    def test_rejects_preload(self):
        with tempfile.NamedTemporaryFile() as library:
            with patch.dict(os.environ, {'LD_PRELOAD': 'unexpected.so'}):
                with self.assertRaises(RuntimeError):
                    load_private_opapi(library.name)

    def test_matching_symbols_and_collision(self):
        class Library:
            aclnnDispatchFFNCombine = ctypes.c_void_p(1)
            aclnnDispatchFFNCombineGetWorkspaceSize = ctypes.c_void_p(2)

        class Collision(Library):
            aclnnDispatchFFNCombine = ctypes.c_void_p(3)

        with tempfile.NamedTemporaryFile() as library:
            with patch.dict(os.environ, {'LD_PRELOAD': ''}):
                with patch('fuseep_a2_loader.ctypes.CDLL', side_effect=[Library(), Library()]) as cdll:
                    load_private_opapi(library.name)
                    self.assertEqual(cdll.call_args_list[0].args, (str(Path(library.name)),))
                    self.assertEqual(cdll.call_args_list[0].kwargs['mode'], os.RTLD_LOCAL | os.RTLD_NOW)
                with patch('fuseep_a2_loader.ctypes.CDLL', side_effect=[Library(), Collision()]):
                    with self.assertRaisesRegex(RuntimeError, 'collision'):
                        load_private_opapi(library.name)


if __name__ == '__main__':
    unittest.main()
