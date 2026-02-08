import unittest

from crystallm.prompt_sanitizer import sanitize_prompt_lines


class TestPromptSanitizer(unittest.TestCase):
    def test_sanitizes_atom_site_injection(self):
        fallback = ["data_Na2Cl2", "_symmetry_space_group_name_H-M   P4/mmm"]
        injected = [
            "data_Na2Cl2",
            "_symmetry_space_group_name_H-M   P4/mmm",
            "_atom_site_label   _atom_site_fract_x   _atom_site_fract_y   _atom_site_fract_z",
            "Na1 0 0 0",
            "Cl1 0.5 0.5 0.5",
        ]
        out = sanitize_prompt_lines(injected, fallback)
        self.assertEqual(out, fallback)

    def test_requires_data_and_space_group(self):
        fallback = ["data_Na2Cl2", "_symmetry_space_group_name_H-M   P4/mmm"]
        injected = ["_chemical_formula_sum 'Na2 Cl2'"]
        out = sanitize_prompt_lines(injected, fallback)
        self.assertEqual(out, fallback)
