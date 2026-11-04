"""Check the published contracts for encoding loss and runnable Python syntax."""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = (
    "lightning_indexer_v2",
    "lightning_indexer_v2_metadata",
    "sparse_lightning_indexer",
    "compressor_v2",
)
SHA = "e7da18105f08b06e01ef0fff78f439098d483746"


class TestOperatorDocumentation(unittest.TestCase):
    def test_utf8_contracts_and_examples(self):
        for language in ("en", "zh"):
            for operator in OPERATORS:
                path = ROOT / f"docs/api/{language}/operators/attention/{operator}.md"
                with self.subTest(path=path):
                    text = path.read_bytes().decode("utf-8", errors="strict")
                    self.assertNotIn("\ufffd", text)
                    self.assertNotRegex(text, r"\?{2,}")
                    self.assertIn(f"/blob/{SHA}/", text)
                    if language == "zh":
                        self.assertRegex(text, r"[\u4e00-\u9fff]")
                        # Escapes keep the guard independent of shell encodings.
                        for heading in (
                            "\u82af\u7247\u652f\u6301",
                            "\u51fd\u6570\u539f\u578b",
                            "\u529f\u80fd\u8bf4\u660e",
                            "\u53c2\u6570\u4e0e\u8fd4\u56de\u503c",
                            "\u8c03\u7528\u793a\u4f8b",
                        ):
                            self.assertIn("## " + heading, text)
                    for example in re.findall(r"```python\n(.*?)```", text, re.S):
                        ast.parse(example, filename=str(path))


if __name__ == "__main__":
    unittest.main()
