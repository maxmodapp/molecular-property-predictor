from pathlib import Path
from fnmatch import fnmatch
import importlib.util

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_dir = Path(globals().get("SPECPATH", Path.cwd())).resolve()
repo_dir = project_dir.parent
icon_path = project_dir / "app_icon.ico"


def collect_tree(source_root: Path, target_prefix: str, excludes=None):
    excludes = excludes or []
    collected = []
    for source_path in source_root.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_root)
        relative_text = relative_path.as_posix()
        if any(fnmatch(relative_text, pattern) or fnmatch(source_path.name, pattern) for pattern in excludes):
            continue
        target_dir = Path(target_prefix) / relative_path.parent
        collected.append((str(source_path), target_dir.as_posix()))
    return collected


def include_runtime_submodule(module_name: str) -> bool:
    if module_name.startswith("rdkit.sping"):
        return False
    ignored_parts = {"tests", "test"}
    return not any(
        part in ignored_parts or part.startswith("test_") or part.startswith("UnitTest")
        for part in module_name.split(".")
    )


def collect_package_tree(package_name: str):
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return []
    return collect_tree(
        Path(spec.origin).parent,
        package_name.replace(".", "/"),
        excludes=["__pycache__/*", "*.pyc"],
    )


datas = [
    (str(project_dir / "app_icon.ico"), "."),
    (str(project_dir / "jsme_editor_embed_fixed.html"), "."),
    (str(repo_dir / "models" / "logp" / "model_logp.keras"), "models/logp"),
    (str(repo_dir / "models" / "logp" / "logp_norm.npz"), "models/logp"),
]

binaries = []
hiddenimports = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "cairosvg",
    "IPython",
    "IPython.display",
    "numpy",
    "pandas",
    "rdkit",
    "rdkit.Chem.AllChem",
    "rdkit.Chem.PandasTools",
    "rdkit.Chem.PropertyMol",
    "rdkit.Chem.rdFMCS",
    "svgutils",
    "svgutils.transform",
    "torch",
    "torch_cluster",
    "torch_geometric",
    "torch_geometric.data",
    "torch_geometric.loader",
    "torch_geometric.nn",
    "torch_geometric.nn.models",
    "torch_scatter",
    "torch_sparse",
    "torch_spline_conv",
    "tqdm",
]

datas += collect_data_files("PySide6")
datas += collect_data_files("rdkit")
datas += collect_data_files("torch_geometric")
binaries += collect_dynamic_libs("rdkit")
binaries += collect_dynamic_libs("torch")

for package_name in ("rdkit", "torch_geometric", "torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv", "cairosvg", "svgutils"):
    hiddenimports += collect_submodules(package_name, filter=include_runtime_submodule)

for package_name in ("torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv"):
    datas += collect_package_tree(package_name)

pkasolver_root = repo_dir / "external" / "pkasolver"
pkasolver_package = pkasolver_root / "pkasolver"
if pkasolver_package.exists():
    datas += collect_tree(
        pkasolver_package,
        "external/pkasolver/pkasolver",
        excludes=[
            "__pycache__/*",
            "*.pyc",
            "tests/*",
        ],
    )

pka_model_dir = repo_dir / "models" / "pka"
if pka_model_dir.exists():
    datas += collect_tree(
        pka_model_dir,
        "models/pka",
        excludes=["__pycache__/*", "*.pyc"],
    )


a = Analysis(
    ["desktop_app_final.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "sklearn",
        "jedi",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PrediccionPropiedadesQuimicas",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon_path) if icon_path.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PrediccionPropiedadesQuimicas",
)
