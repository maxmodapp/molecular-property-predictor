from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


PKASOLVER_HOME_ENV = "PKASOLVER_HOME"
PKA_REACTION_MIN_HEIGHT = 190
PKA_REACTION_MAX_HEIGHT = 320
MOLECULE_IMAGE_MIN_SIZE = 300
MOLECULE_IMAGE_MAX_SIZE = 560
_DLL_DIRECTORY_HANDLES = []


@dataclass
class PkaSolverStateSummary:
    index: int
    pka: float
    pka_stddev: float
    protonated_smiles: str
    deprotonated_smiles: str


@dataclass
class PkaSolverPrediction:
    states: list[PkaSolverStateSummary]
    reactions_png: bytes | None
    map_png: bytes | None


def _image_to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _adaptive_mol_image_size(
    mol,
    min_size: int = MOLECULE_IMAGE_MIN_SIZE,
    max_size: int = MOLECULE_IMAGE_MAX_SIZE,
) -> tuple[int, int]:
    atom_count = max(1, mol.GetNumAtoms())
    bond_count = mol.GetNumBonds()
    size = 240 + atom_count * 10 + bond_count * 4
    size = max(min_size, min(max_size, size))
    return int(size), int(size)


def _adaptive_pka_reaction_height(protonation_states: list) -> int:
    atom_counts = []
    for state in protonation_states:
        atom_counts.append(state.protonated_mol.GetNumAtoms())
        atom_counts.append(state.deprotonated_mol.GetNumAtoms())
    max_atoms = max(atom_counts, default=1)
    height = 165 + max_atoms * 5
    return int(max(PKA_REACTION_MIN_HEIGHT, min(PKA_REACTION_MAX_HEIGHT, height)))


def _prepare_native_dll_search_path():
    if not sys.platform.startswith("win"):
        return

    candidates = [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
        Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)),
    ]

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_text = str(candidate)
        if candidate_text not in path_parts:
            os.environ["PATH"] = candidate_text + os.pathsep + os.environ.get("PATH", "")
            path_parts.insert(0, candidate_text)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(candidate_text))
            except OSError:
                pass


class PkaSolverPredictor:
    """Project-owned adapter around pKaSolver's public query functions.

    This class is intentionally thin: pKaSolver remains the pKa engine, while
    the application owns path discovery, model caching, PNG conversion, and the
    shape of the data returned to the UI.
    """

    def __init__(self, package_root: Path | None, error_message: str = ""):
        self.package_root = package_root.resolve() if package_root else None
        self.error_message = error_message
        self._modules: dict[str, object] | None = None
        self._query_model = None
        self._lock = threading.Lock()

    @classmethod
    def discover(cls, base_dir: Path, source_dir: Path | None = None):
        candidates: list[Path] = []

        env_home = os.environ.get(PKASOLVER_HOME_ENV, "").strip()
        if env_home:
            candidates.append(Path(env_home))

        adapter_source_dir = Path(__file__).resolve().parent
        if source_dir is None:
            source_dir = adapter_source_dir

        candidates.extend([
            base_dir / "external" / "pkasolver",
            base_dir.parent / "external" / "pkasolver",
            source_dir / "external" / "pkasolver",
            source_dir.parent / "external" / "pkasolver",
            adapter_source_dir / "external" / "pkasolver",
            adapter_source_dir.parent / "external" / "pkasolver",
        ])

        seen: set[Path] = set()
        for candidate in candidates:
            root = candidate.resolve()
            if root in seen:
                continue
            seen.add(root)
            if (root / "pkasolver" / "query.py").exists():
                return cls(root)

        if importlib.util.find_spec("pkasolver") is not None:
            return cls(None)

        return cls(
            None,
            "pKaSolver no esta configurado. Deja external/pkasolver junto a app o define PKASOLVER_HOME.",
        )

    def is_available(self) -> bool:
        return self.package_root is not None or importlib.util.find_spec("pkasolver") is not None

    def predict(self, smiles: str) -> PkaSolverPrediction:
        with self._lock:
            modules = self._ensure_modules()
            query_model = self._ensure_query_model(modules)
            chem = modules["Chem"]
            calculate_microstate_pka_values = modules["calculate_microstate_pka_values"]
            draw_pka_reactions = modules["draw_pka_reactions"]
            draw_pka_map = modules["draw_pka_map"]
            cairosvg = modules["cairosvg"]

            mol = chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("pKaSolver recibio un SMILES invalido.")

            with contextlib.redirect_stdout(io.StringIO()):
                protonation_states = calculate_microstate_pka_values(mol, query_model=query_model)

            summaries = [
                PkaSolverStateSummary(
                    index=index,
                    pka=float(state.pka),
                    pka_stddev=float(state.pka_stddev),
                    protonated_smiles=chem.MolToSmiles(state.protonated_mol, canonical=True),
                    deprotonated_smiles=chem.MolToSmiles(state.deprotonated_mol, canonical=True),
                )
                for index, state in enumerate(protonation_states)
            ]

            if not protonation_states:
                return PkaSolverPrediction(summaries, reactions_png=None, map_png=None)

            reactions_png = self._render_reactions_png(
                draw_pka_reactions,
                cairosvg,
                protonation_states,
            )
            map_size = _adaptive_mol_image_size(protonation_states[0].ph7_mol)
            map_png = _image_to_png_bytes(draw_pka_map(protonation_states, size=map_size))
            return PkaSolverPrediction(summaries, reactions_png=reactions_png, map_png=map_png)

    def _ensure_modules(self) -> dict[str, object]:
        if self._modules is not None:
            return self._modules

        if self.package_root:
            package_root_text = str(self.package_root)
            if package_root_text not in sys.path:
                sys.path.insert(0, package_root_text)

        _prepare_native_dll_search_path()

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                from rdkit import Chem as PkaChem
                from pkasolver.query import (
                    QueryModel,
                    calculate_microstate_pka_values,
                    draw_pka_map,
                    draw_pka_reactions,
                )
                import cairosvg
        except Exception as exc:
            raise RuntimeError(
                "No se pudo cargar pKaSolver. Revisa que el entorno tenga rdkit, torch, "
                f"torch_geometric, svgutils, cairosvg e IPython. Detalle: {exc}"
            ) from exc

        self._modules = {
            "Chem": PkaChem,
            "QueryModel": QueryModel,
            "calculate_microstate_pka_values": calculate_microstate_pka_values,
            "draw_pka_map": draw_pka_map,
            "draw_pka_reactions": draw_pka_reactions,
            "cairosvg": cairosvg,
        }
        return self._modules

    def _ensure_query_model(self, modules: dict[str, object]):
        if self._query_model is None:
            query_model_class = modules["QueryModel"]
            self._query_model = query_model_class()
        return self._query_model

    def _render_reactions_png(self, draw_pka_reactions, cairosvg, protonation_states: list) -> bytes:
        with contextlib.redirect_stdout(io.StringIO()):
            svg = draw_pka_reactions(
                protonation_states,
                height=_adaptive_pka_reaction_height(protonation_states),
            )
        svg_text = getattr(svg, "data", None)
        if svg_text is None:
            svg_text = str(svg)
        if isinstance(svg_text, bytes):
            svg_bytes = svg_text
        else:
            svg_bytes = svg_text.encode("utf-8")
        return cairosvg.svg2png(bytestring=svg_bytes)
