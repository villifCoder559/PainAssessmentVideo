import ast
import unittest
from pathlib import Path


class TestTrainModelCliHelp(unittest.TestCase):
    def test_scheduler_name_help_lists_all_learning_rate_schedulers(self):
        source_path = Path(__file__).resolve().parents[1] / "train_model.py"
        tree = ast.parse(source_path.read_text(), filename=str(source_path))

        scheduler_argument = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--scheduler_name"
        )
        help_text = next(
            keyword.value.value
            for keyword in scheduler_argument.keywords
            if keyword.arg == "help" and isinstance(keyword.value, ast.Constant)
        )

        self.assertEqual(
            help_text,
            "Learning-rate scheduler. Choices: cosine, cosine_restart, step, onecycle, lambda. "
            "Default: cosine.",
        )


if __name__ == "__main__":
    unittest.main()
