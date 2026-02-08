import unittest

from crystallm._configuration import _coerce_value


class TestConfigurationParsing(unittest.TestCase):
    def test_coerce_bool(self):
        for raw in ["1", "true", "TRUE", "yes", "on", "Y"]:
            self.assertTrue(_coerce_value(bool, raw))
        for raw in ["0", "false", "FALSE", "no", "off", "N"]:
            self.assertFalse(_coerce_value(bool, raw))

        with self.assertRaises(ValueError):
            _coerce_value(bool, "maybe")

