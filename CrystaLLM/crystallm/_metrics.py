import re
import traceback

try:
    import spglib
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: spglib. Install it in your current Python environment, e.g.:\n"
        "  python -m pip install spglib==2.6.0\n"
        "or (conda):\n"
        "  conda install -c conda-forge spglib\n"
        "CrystaLLM's CIF validation relies on spglib for symmetry checks."
    ) from exc
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ._utils import extract_data_formula


def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            j = connected_site_info['site_index']
            if i == j:  # skip if they're the same site
                continue
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

            is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

            electronegativity_diff = abs(site.specie.X - connected_site.specie.X)
            """
            According to the Pauling scale, when the electronegativity difference 
            between two bonded atoms is less than 1.7, the bond can be considered 
            to have predominantly covalent character, while a difference greater 
            than or equal to 1.7 indicates that the bond has significant ionic 
            character.
            """
            expected_length = None
            try:
                if electronegativity_diff >= 1.7:
                    # use ionic radii
                    if site.specie.X < connected_site.specie.X:
                        expected_length = site.specie.average_cationic_radius + connected_site.specie.average_anionic_radius
                    else:
                        expected_length = site.specie.average_anionic_radius + connected_site.specie.average_cationic_radius
                else:
                    expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius
            except Exception:  # noqa: BLE001
                expected_length = None

            # If we cannot determine a meaningful expected length, skip this edge.
            if expected_length is None:
                continue
            try:
                expected_length = float(expected_length)
            except Exception:  # noqa: BLE001
                continue
            if expected_length <= 0:
                continue

            bond_ratio = bond_length / expected_length

            # penalize bond lengths that are too short or too long;
            #  check if bond involves hydrogen and adjust tolerance accordingly
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    # If CrystalNN finds no bonds (common for malformed/degenerate structures),
    # treat as maximally unreasonable instead of throwing ZeroDivisionError.
    if bond_count <= 0:
        return 0.0

    normalized_score = score / bond_count

    return normalized_score


def is_space_group_consistent(cif_str):
    """
    Backwards-compatible boolean wrapper.

    Prefer using space_group_consistency_details() if you need to distinguish:
    - mismatch (False)
    - analysis/parse failure (None)
    """
    details = space_group_consistency_details(cif_str)
    return details.get("consistent") is True


def space_group_consistency_details(cif_str, symprec: float = 0.1):
    """
    Try to check space group consistency.

    Returns a dict with:
      - consistent: True / False / None
        - True: stated == detected
        - False: stated and detected both available, but mismatch
        - None: failed to parse/analyze, so cannot decide
      - stated_space_group: str | None
      - detected_space_group: str | None
      - error: str | None
      - traceback: str | None
    """
    # 1) Parse structure
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
    except Exception as exc:  # noqa: BLE001
        return {
            "consistent": None,
            "stated_space_group": None,
            "detected_space_group": None,
            "error": f"parse_structure: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # 2) Extract stated SG (best-effort)
    stated_space_group = None
    try:
        parser = CifParser.from_string(cif_str)
        cif_data = parser.as_dict()
        block = list(cif_data.keys())[0]
        stated_space_group = cif_data[block].get("_symmetry_space_group_name_H-M")
        # Some CIFs may store values as list-like.
        if isinstance(stated_space_group, list):
            stated_space_group = stated_space_group[0] if stated_space_group else None
        if stated_space_group is None:
            return {
                "consistent": None,
                "stated_space_group": None,
                "detected_space_group": None,
                "error": "missing_stated_space_group",
                "traceback": None,
            }
        stated_space_group = str(stated_space_group).strip().strip("'\"")
        if not stated_space_group:
            return {
                "consistent": None,
                "stated_space_group": None,
                "detected_space_group": None,
                "error": "empty_stated_space_group",
                "traceback": None,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "consistent": None,
            "stated_space_group": None,
            "detected_space_group": None,
            "error": f"parse_stated_space_group: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # 3) Analyze symmetry (may fail for degenerate/bad structures)
    try:
        def _spglib_diagnostics(structure_):
            # Best-effort: fetch a more specific spglib error and quick diagnostics.
            # Note: spglib does not always raise; it may return None and set an internal error message.
            spglib_err_ = None
            try:
                cell = (
                    structure_.lattice.matrix,
                    structure_.frac_coords,
                    [site.specie.Z for site in structure_],
                )
                _ = spglib.get_symmetry_dataset(cell, symprec=symprec)
                spglib_err_ = spglib.get_error_message()
            except Exception:  # noqa: BLE001
                spglib_err_ = None

            min_neighbor_dist_ = None
            try:
                # Only search within symprec; if spglib complains about "too close",
                # this helps surface the actual scale of the collision.
                _, _, _, dists = structure_.get_neighbor_list(r=symprec)
                if len(dists) > 0:
                    min_neighbor_dist_ = float(min(dists))
            except Exception:  # noqa: BLE001
                min_neighbor_dist_ = None

            return spglib_err_, min_neighbor_dist_

        spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        dataset = spacegroup_analyzer.get_symmetry_dataset()
        recovered_from_too_close = False

        if not dataset:
            spglib_err, min_neighbor_dist = _spglib_diagnostics(structure)

            # Recovery path: the most common failure mode in our pipeline is that
            # symmetry op replacement expands to many nearly-duplicate sites
            # (spglib: "too close distance between atoms"). Merging sites at the
            # same tolerance often restores a usable symmetry dataset.
            if spglib_err and "too close distance" in spglib_err.lower():
                try:
                    merged = structure.copy()
                    merged.merge_sites(tol=symprec, mode="delete")
                    dataset = SpacegroupAnalyzer(merged, symprec=symprec).get_symmetry_dataset()
                    if dataset:
                        structure = merged
                        recovered_from_too_close = True
                except Exception:  # noqa: BLE001
                    dataset = None

            if not dataset:
                err_parts = ["symmetry_dataset_unavailable"]
                if spglib_err and spglib_err.strip() and spglib_err.strip().lower() != "no error":
                    err_parts.append(spglib_err.strip())
                if min_neighbor_dist is not None:
                    err_parts.append(f"min_neighbor_dist<{symprec:g}Å≈{min_neighbor_dist:.4f}Å")
                err = ": ".join(err_parts)
                return {
                    "consistent": None,
                    "stated_space_group": stated_space_group,
                    "detected_space_group": None,
                    "error": err,
                    "traceback": None,
                }

        # pymatgen may return a dict-like OR a spglib dataset object.
        # Newer pymatgen/spglib deprecate dict-style access for dataset objects and
        # recommend the attribute interface, otherwise you get:
        #   DeprecationWarning: dict interface is deprecated. Use attribute interface instead
        detected_space_group = getattr(dataset, "international", None) or getattr(dataset, "international_symbol", None)
        if detected_space_group is None and isinstance(dataset, dict):
            detected_space_group = dataset.get("international") or dataset.get("international_symbol")

        if detected_space_group is None:
            return {
                "consistent": None,
                "stated_space_group": stated_space_group,
                "detected_space_group": None,
                "error": "detected_space_group_missing",
                "traceback": None,
            }
        detected_space_group = str(detected_space_group).strip().strip("'\"")
        if not detected_space_group:
            return {
                "consistent": None,
                "stated_space_group": stated_space_group,
                "detected_space_group": None,
                "error": "detected_space_group_empty",
                "traceback": None,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "consistent": None,
            "stated_space_group": stated_space_group,
            "detected_space_group": None,
            "error": f"symmetry_analysis_failed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    return {
        "consistent": stated_space_group.strip() == detected_space_group.strip(),
        "stated_space_group": stated_space_group,
        "detected_space_group": detected_space_group,
        "error": None,
        "traceback": None,
        "recovered_from_too_close_distance": recovered_from_too_close,
    }


def is_formula_consistent(cif_str):
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    formula_data = Composition(extract_data_formula(cif_str))
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])

    return formula_data.reduced_formula == formula_sum.reduced_formula == formula_structural.reduced_formula


def is_atom_site_multiplicity_consistent(cif_str):
    # Parse the CIF string
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    # Extract the chemical formula sum from the CIF data
    formula_sum = cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"]

    # Convert the formula sum into a dictionary
    expected_atoms = Composition(formula_sum).as_dict()

    # Count the atoms provided in the _atom_site_type_symbol section
    actual_atoms = {}
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):
                if atom_type in actual_atoms:
                    actual_atoms[atom_type] += int(multiplicity)
                else:
                    actual_atoms[atom_type] = int(multiplicity)

    # Validate if the expected and actual atom counts match
    return expected_atoms == actual_atoms


def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True


def is_valid(cif_str, bond_length_acceptability_cutoff=1.0, check_composition: bool = True):
    if check_composition and not is_formula_consistent(cif_str):
        return False
    if check_composition and not is_atom_site_multiplicity_consistent(cif_str):
        return False
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score < bond_length_acceptability_cutoff:
        return False
    if not is_space_group_consistent(cif_str):
        return False
    return True
