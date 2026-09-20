from __future__ import annotations

import ctypes
import html
import io
import json
import os
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

from PySide6.QtCore import QThread, Qt, QUrl, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView

from pka_prediction_engine import (
    PkaSolverPrediction,
    PkaSolverPredictor,
    PkaSolverStateSummary,
)


os.environ.setdefault("PYTORCH_JIT", "0")

SOURCE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_DIR))
RESOURCE_ROOT = APP_DIR if getattr(sys, "frozen", False) else SOURCE_DIR.parent

MODEL_LOGP_PATH = RESOURCE_ROOT / "models" / "logp" / "model_logp.keras"
NORM_LOGP_PATH = RESOURCE_ROOT / "models" / "logp" / "logp_norm.npz"
EDITOR_HTML_PATH = APP_DIR / "jsme_editor_embed_fixed.html"
ICON_PATH = APP_DIR / "app_icon.ico"
APP_USER_MODEL_ID = "maxim.prediccion.propiedades.quimicas"
PKASOLVER_SMOKE_TEST_ARG = "--smoke-test-pkasolver"

TOP_SECTION_HEIGHT = 520
EDITOR_WEB_HEIGHT = 320
BOTTOM_PANEL_MIN_HEIGHT = 260
IMAGE_PADDING = 24
MOLECULE_IMAGE_MIN_SIZE = 300
MOLECULE_IMAGE_MAX_SIZE = 560
PKA_SMILES_FONT_SIZE = 13
PKA_VALUE_FONT_SIZE = 15
PKA_BLOCK_SPACING = 20


def load_norm_params(npz_path: Path) -> tuple[float, float]:
    data = np.load(npz_path, allow_pickle=True)
    mean = None
    std = None
    for key in ["y_mean", "mean", "target_mean"]:
        if key in data.files:
            mean = float(np.asarray(data[key]).reshape(-1)[0])
            break
    for key in ["y_std", "std", "target_std"]:
        if key in data.files:
            std = float(np.asarray(data[key]).reshape(-1)[0])
            break
    if mean is None or std is None:
        raise ValueError(f"No pude encontrar mean/std en {npz_path.name}. Claves: {data.files}")
    return mean, std


def denorm(value_norm: np.ndarray | float, mean: float, std: float) -> float:
    value = float(np.asarray(value_norm).reshape(-1)[0])
    return value * std + mean


def normalize_inchi_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Ingresa un InChI o un SMILES.")
    if text.startswith("InChI="):
        return text
    if text.startswith(("1S/", "1/")):
        return "InChI=" + text
    return text


def parse_structure(input_text: str):
    text = normalize_inchi_text(input_text)

    if text.startswith("InChI="):
        mol = Chem.MolFromInchi(text)
        if mol is None:
            raise ValueError("El InChI no pudo convertirse en una molecula valida.")
        return mol, "InChI"

    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return mol, "SMILES"

    maybe_inchi = "InChI=" + text
    mol = Chem.MolFromInchi(maybe_inchi)
    if mol is not None:
        return mol, "InChI"

    raise ValueError("No pude interpretar la entrada como InChI ni como SMILES.")


def mol_to_smiles(mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_inchi(mol) -> str:
    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return "No disponible"


def format_inchi_display(inchi: str) -> str:
    if inchi.startswith("InChI="):
        return inchi[len("InChI="):]
    return inchi


def mol_to_fp_array(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)


def image_to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def adaptive_mol_image_size(
    mol,
    min_size: int = MOLECULE_IMAGE_MIN_SIZE,
    max_size: int = MOLECULE_IMAGE_MAX_SIZE,
) -> tuple[int, int]:
    atom_count = max(1, mol.GetNumAtoms())
    bond_count = mol.GetNumBonds()
    size = 240 + atom_count * 10 + bond_count * 4
    size = max(min_size, min(max_size, size))
    return int(size), int(size)


def mol_to_png_bytes(mol, size: tuple[int, int] | None = None) -> bytes:
    if size is None:
        size = adaptive_mol_image_size(mol)
    img = Draw.MolToImage(mol, size=size)
    return image_to_png_bytes(img)


def show_pixmap_at_natural_size(label: QLabel, pixmap: QPixmap, minimum_height: int):
    label.setPixmap(pixmap)
    label.setMinimumSize(
        pixmap.width() + IMAGE_PADDING,
        max(minimum_height, pixmap.height() + IMAGE_PADDING),
    )


def enable_label_text_selection(label: QLabel):
    label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
    label.setFocusPolicy(Qt.StrongFocus)


class PropertyPredictor:
    def __init__(self, model_path: Path, norm_path: Path, name: str):
        self.name = name
        if tf is None:
            raise RuntimeError("TensorFlow no esta instalado en este entorno.")
        if not model_path.exists():
            raise FileNotFoundError(f"No encontre el modelo: {model_path.name}")
        if not norm_path.exists():
            raise FileNotFoundError(f"No encontre el archivo de normalizacion: {norm_path.name}")
        self.model = self._load_model(model_path)
        self.y_mean, self.y_std = load_norm_params(norm_path)

    def predict_from_fp(self, fp_batch: np.ndarray) -> float:
        pred_norm = self.model.predict(fp_batch, verbose=0)
        return denorm(pred_norm, self.y_mean, self.y_std)

    def _load_model(self, model_path: Path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except TypeError as exc:
            if model_path.suffix.lower() != ".keras" or "quantization_config" not in str(exc):
                raise
            return self._load_sanitized_keras_archive(model_path)

    def _load_sanitized_keras_archive(self, model_path: Path):
        with tempfile.TemporaryDirectory() as temp_dir:
            sanitized_path = Path(temp_dir) / model_path.name
            self._write_sanitized_keras_archive(model_path, sanitized_path)
            return tf.keras.models.load_model(sanitized_path, compile=False)

    def _write_sanitized_keras_archive(self, source_path: Path, target_path: Path):
        with zipfile.ZipFile(source_path, "r") as source_zip:
            with zipfile.ZipFile(target_path, "w") as target_zip:
                for item in source_zip.infolist():
                    content = source_zip.read(item.filename)
                    if item.filename == "config.json":
                        config = json.loads(content.decode("utf-8"))
                        config = self._remove_unsupported_keras_keys(config)
                        content = json.dumps(config, ensure_ascii=False).encode("utf-8")
                    target_zip.writestr(item, content)

    def _remove_unsupported_keras_keys(self, value):
        if isinstance(value, dict):
            return {
                key: self._remove_unsupported_keras_keys(child)
                for key, child in value.items()
                if key != "quantization_config"
            }
        if isinstance(value, list):
            return [self._remove_unsupported_keras_keys(child) for child in value]
        return value


class PkaSolverWorker(QThread):
    succeeded = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, request_id: int, smiles: str, predictor: PkaSolverPredictor):
        super().__init__()
        self.request_id = request_id
        self.smiles = smiles
        self.predictor = predictor

    def run(self):
        if self.isInterruptionRequested():
            return
        try:
            prediction = self.predictor.predict(self.smiles)
            if not self.isInterruptionRequested():
                self.succeeded.emit(self.request_id, prediction)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(self.request_id, str(exc))


class PredictionSummaryCard(QFrame):
    property_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("predictionCardOuter")
        self.setStyleSheet(
            """
            QFrame#predictionCardOuter {
                background: transparent;
                border: none;
            }
            QFrame#predictionCardPanel {
                background-color: #1f2430;
                border: 1px solid #3a4151;
                border-radius: 12px;
            }
            QLabel { color: #f5f7fa; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._property_tab_keys = ["pka", "logp"]
        self.property_tabs = QTabBar()
        self.property_tabs.addTab("pKa")
        self.property_tabs.addTab("log P")
        self.property_tabs.setExpanding(False)
        self.property_tabs.setDrawBase(False)
        self.property_tabs.setStyleSheet(
            """
            QTabBar::tab {
                background: #161b22;
                color: #f5f7fa;
                padding: 8px 16px;
                margin-right: 4px;
                border: 1px solid #3a4151;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #2f81f7;
                color: #ffffff;
            }
            """
        )

        tab_row = QWidget()
        tab_row.setStyleSheet("background: transparent;")
        tab_layout = QHBoxLayout(tab_row)
        tab_layout.setContentsMargins(12, 0, 0, 0)
        tab_layout.setSpacing(0)
        tab_layout.addWidget(self.property_tabs)
        tab_layout.addStretch(1)

        panel = QFrame()
        panel.setObjectName("predictionCardPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        title = QLabel("Predicciones")
        title.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #9ecbff; "
            "background-color: #1f2430; border: none;"
        )

        self.pka_tab = QWidget()
        self.pka_tab.setStyleSheet("background-color: #1f2430;")
        pka_layout = QVBoxLayout(self.pka_tab)
        pka_layout.setContentsMargins(8, 10, 8, 8)
        pka_layout.setSpacing(8)

        self.logp_tab = QWidget()
        self.logp_tab.setStyleSheet("background-color: #1f2430;")
        logp_layout = QVBoxLayout(self.logp_tab)
        logp_layout.setContentsMargins(8, 10, 8, 8)
        logp_layout.setSpacing(8)

        self.logp_box = QFrame()
        self.logp_box.setObjectName("logpPredictionBox")
        self.logp_box.setStyleSheet(
            """
            QFrame#logpPredictionBox {
                color: #ffffff;
                background-color: #1f2430;
                border: 1px solid #3a4151;
                border-radius: 8px;
            }
            """
        )
        logp_box_layout = QHBoxLayout(self.logp_box)
        logp_box_layout.setContentsMargins(10, 6, 10, 6)
        logp_box_layout.setSpacing(4)

        self.logp_name = QLabel("log P =")
        self.logp_name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.logp_name.setStyleSheet(
            "font-size: 13px; font-weight: 800; color: #c9d1d9; "
            "background-color: #1f2430; border: none;"
        )
        enable_label_text_selection(self.logp_name)

        self.logp_value = QLabel("-")
        self.logp_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.logp_value.setStyleSheet(
            "font-size: 24px; font-weight: 800; color: #ffffff; "
            "background-color: #1f2430; border: none;"
        )
        enable_label_text_selection(self.logp_value)

        logp_box_layout.addWidget(self.logp_name, 0, Qt.AlignVCenter)
        logp_box_layout.addWidget(self.logp_value, 1, Qt.AlignVCenter)

        self.pka_status = QLabel("Esperando entrada")
        self.pka_status.setWordWrap(True)
        self.pka_status.setStyleSheet("font-size: 13px; color: #c9d1d9;")
        enable_label_text_selection(self.pka_status)

        self.pka_values = QLabel("-")
        self.pka_values.setWordWrap(True)
        self.pka_values.setTextFormat(Qt.RichText)
        enable_label_text_selection(self.pka_values)
        self.pka_values.setStyleSheet(
            """
            QLabel {
                font-size: 13px;
                color: #ffffff;
                background-color: #1f2430;
                border: 1px solid #3a4151;
                border-radius: 8px;
                padding: 10px;
            }
            """
        )

        panel_layout.addWidget(title)
        pka_layout.addWidget(self.pka_status)
        pka_layout.addWidget(self.pka_values)
        pka_layout.addStretch(1)

        logp_layout.addWidget(self.logp_box)
        logp_layout.addStretch(1)

        panel_layout.addWidget(self.pka_tab)
        panel_layout.addWidget(self.logp_tab)
        layout.addWidget(tab_row)
        layout.addWidget(panel)
        self.property_tabs.currentChanged.connect(self._handle_property_tab_changed)
        self.set_logp_value("-")
        self.set_selected_property("pka")

    def selected_property(self) -> str:
        index = self.property_tabs.currentIndex()
        if 0 <= index < len(self._property_tab_keys):
            return self._property_tab_keys[index]
        return "pka"

    def _handle_property_tab_changed(self, index: int):
        property_key = self.selected_property()
        self.set_selected_property(property_key)
        self.property_changed.emit(property_key)

    def set_selected_property(self, property_key: str):
        is_pka = property_key == "pka"
        self.pka_tab.setVisible(is_pka)
        self.logp_tab.setVisible(not is_pka)

    def _build_metric_row(self, label_text: str, value_label: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        label = QLabel(label_text)
        label.setStyleSheet("font-size: 13px; font-weight: 700; color: #c9d1d9;")
        label.setAlignment(Qt.AlignTop)

        row.addWidget(label, 0, Qt.AlignTop)
        row.addWidget(value_label, 1)
        return row

    def reset(self):
        self.set_logp_value("-")
        self.set_pka_idle()

    def set_logp_value(self, value_text: str):
        self.logp_value.setText(value_text)

    def set_pka_idle(self):
        self.pka_status.setText("Esperando entrada")
        self.pka_status.setStyleSheet("font-size: 13px; color: #c9d1d9;")
        self.pka_values.setText("-")

    def set_pka_pending(self, message: str = "Calculando pKa de los microestados..."):
        self.pka_status.setText(message)
        self.pka_status.setStyleSheet("font-size: 13px; color: #9ecbff;")
        self.pka_values.setText("Calculando...")

    def set_pka_result(self, prediction: PkaSolverPrediction):
        if not prediction.states:
            self.pka_status.setText("No se detectaron pKa de microestados")
            self.pka_status.setStyleSheet("font-size: 13px; color: #ffb86b;")
            self.pka_values.setText("No hay microestados ionizables para listar.")
            return

        count = len(prediction.states)
        if count == 1:
            self.pka_status.setText("Calculado 1 pKa de microestado")
        else:
            self.pka_status.setText(f"Calculados {count} pKa de microestados")
        self.pka_status.setStyleSheet("font-size: 13px; color: #7ee787;")
        self.pka_values.setText(self._format_pka_values(prediction.states))

    def set_pka_error(self, message: str):
        self.pka_status.setText(message)
        self.pka_status.setStyleSheet("font-size: 13px; color: #ffb86b;")
        self.pka_values.setText("No disponible")

    def _format_pka_values(self, states: list[PkaSolverStateSummary]) -> str:
        blocks = []
        for state in states:
            low = state.pka - state.pka_stddev
            high = state.pka + state.pka_stddev
            protonated = html.escape(state.protonated_smiles)
            deprotonated = html.escape(state.deprotonated_smiles)
            blocks.append(
                (
                    "<div>"
                    f"<div style='font-size: {PKA_SMILES_FONT_SIZE}px; "
                    "font-weight: 600; color: #f5f7fa; line-height: 1.35;'>"
                    f"{protonated} &#8646; {deprotonated}"
                    "</div>"
                    f"<div style='font-size: {PKA_VALUE_FONT_SIZE}px; "
                    "color: #ffffff; line-height: 1.35;'>"
                    f"<span style='font-weight: 800;'>pKa {state.index + 1} = {state.pka:.2f}</span> "
                    f"<span style='font-weight: 500;'>({low:.2f}; {high:.2f})</span>"
                    "</div>"
                    "</div>"
                )
            )
        if PKA_BLOCK_SPACING <= 0:
            return "".join(blocks)
        separator = f"<div style='font-size: {PKA_BLOCK_SPACING}px; color: #1f2430;'>&nbsp;</div>"
        return separator.join(blocks)


class ResultsPanel(QFrame):
    property_changed = Signal(str)

    def __init__(self, logp_predictor):
        super().__init__()
        self.logp_predictor = logp_predictor
        self.setStyleSheet(
            """
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            QLabel { color: #f5f7fa; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("Resultados")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #9ecbff;")

        self.status_label = QLabel("Estado: esperando entrada")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #c9d1d9;")
        enable_label_text_selection(self.status_label)

        self.smiles_label = QLabel("SMILES: -")
        self.smiles_label.setWordWrap(True)
        self.smiles_label.setStyleSheet("font-size: 13px; color: #f5f7fa;")
        enable_label_text_selection(self.smiles_label)

        self.inchi_label = QLabel("InChI: -")
        self.inchi_label.setWordWrap(True)
        self.inchi_label.setStyleSheet("font-size: 13px; color: #f5f7fa;")
        enable_label_text_selection(self.inchi_label)

        self.prediction_card = PredictionSummaryCard()
        self.prediction_card.property_changed.connect(self.property_changed.emit)

        layout.addWidget(title)
        layout.addWidget(self.status_label)
        layout.addWidget(self.smiles_label)
        layout.addWidget(self.inchi_label)
        layout.addWidget(self.prediction_card)
        layout.addStretch(1)

        self.setMinimumHeight(BOTTOM_PANEL_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.prediction_card.set_selected_property(self.selected_property())

    def selected_property(self) -> str:
        return self.prediction_card.selected_property()

    def reset(self):
        self.status_label.setText("Estado: esperando entrada")
        self.smiles_label.setText("SMILES: -")
        self.inchi_label.setText("InChI: -")
        self.prediction_card.reset()

    def update_for_molecule(self, mol, source_kind: str):
        smiles = mol_to_smiles(mol)
        inchi = format_inchi_display(mol_to_inchi(mol))
        fp = mol_to_fp_array(mol)

        self.status_label.setText(f"Estado: molecula valida desde {source_kind}.")
        self.smiles_label.setText(f"SMILES: {smiles}")
        self.inchi_label.setText(f"InChI: {inchi}")

        if self.logp_predictor:
            logp_text = f"{self.logp_predictor.predict_from_fp(fp):.4f}"
        else:
            logp_text = "No cargado"
        self.prediction_card.set_logp_value(logp_text)
        self.prediction_card.set_pka_idle()

    def set_pka_pending(self, message: str = "Calculando pKa de los microestados..."):
        self.prediction_card.set_pka_pending(message)

    def set_pka_result(self, prediction: PkaSolverPrediction):
        self.prediction_card.set_pka_result(prediction)

    def set_pka_error(self, message: str):
        self.prediction_card.set_pka_error(message)


class MoleculeViewer(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            """
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            QLabel { color: #f5f7fa; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Molecula")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #9ecbff;")

        self.image_label = self._build_image_label("La molecula se mostrara aca", BOTTOM_PANEL_MIN_HEIGHT)

        self._molecule_pixmap = None

        layout.addWidget(title)
        layout.addWidget(self.image_label)

        self.setMinimumHeight(BOTTOM_PANEL_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _build_image_label(self, placeholder: str, min_height: int) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(min_height)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                color: #666;
                border-radius: 10px;
                padding: 8px;
            }
            """
        )
        return label

    def clear(self):
        self._molecule_pixmap = None
        self._set_placeholder(self.image_label, "La molecula se mostrara aca", BOTTOM_PANEL_MIN_HEIGHT)

    def update_for_molecule(self, mol):
        png_bytes = mol_to_png_bytes(mol)
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes, "PNG")
        self._molecule_pixmap = pixmap
        self.image_label.setText("")
        self._refresh_pixmap()

    def _set_placeholder(self, label: QLabel, text: str, min_height: int):
        label.setPixmap(QPixmap())
        label.setText(text)
        label.setMinimumSize(0, min_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        self._refresh_label_pixmap(
            self.image_label,
            self._molecule_pixmap,
            minimum_height=BOTTOM_PANEL_MIN_HEIGHT,
        )

    def _refresh_label_pixmap(
        self,
        label: QLabel,
        pixmap: QPixmap | None,
        minimum_height: int,
    ):
        if not pixmap or pixmap.isNull():
            return

        show_pixmap_at_natural_size(label, pixmap, minimum_height)


class PkaImagesPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            """
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            QLabel { color: #f5f7fa; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Microestados y mapa de pKa")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #9ecbff;")

        reactions_title = QLabel("Reacciones de microestados")
        reactions_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #c9d1d9;")
        self.reactions_label = self._build_image_label("Sin calculo de microestados", 220)

        map_title = QLabel("Mapa total")
        map_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #c9d1d9;")
        self.map_label = self._build_image_label("Sin mapa total", 260)

        self._reactions_pixmap = None
        self._map_pixmap = None

        layout.addWidget(title)
        layout.addWidget(reactions_title)
        layout.addWidget(self.reactions_label)
        layout.addWidget(map_title)
        layout.addWidget(self.map_label)

    def _build_image_label(self, placeholder: str, min_height: int) -> QLabel:
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(min_height)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                color: #666;
                border-radius: 10px;
                padding: 8px;
            }
            """
        )
        return label

    def clear(self):
        self._reactions_pixmap = None
        self._map_pixmap = None
        self._set_placeholder(self.reactions_label, "Sin calculo de microestados", 220)
        self._set_placeholder(self.map_label, "Sin mapa total", 260)

    def set_pending(self):
        self._reactions_pixmap = None
        self._map_pixmap = None
        self._set_placeholder(self.reactions_label, "Calculando microestados", 220)
        self._set_placeholder(self.map_label, "Calculando mapa total", 260)

    def set_result(self, prediction: PkaSolverPrediction):
        if not prediction.states:
            self._reactions_pixmap = None
            self._map_pixmap = None
            self._set_placeholder(self.reactions_label, "Sin microestados para mostrar", 220)
            self._set_placeholder(self.map_label, "Sin mapa total para mostrar", 260)
            return

        self._reactions_pixmap = self._pixmap_from_png(prediction.reactions_png)
        self._map_pixmap = self._pixmap_from_png(prediction.map_png)
        self.reactions_label.setText("")
        self.map_label.setText("")
        self._refresh_pixmap()

    def set_error(self, message: str):
        self._reactions_pixmap = None
        self._map_pixmap = None
        self._set_placeholder(self.reactions_label, "Microestados no disponibles", 220)
        self._set_placeholder(self.map_label, "Mapa total no disponible", 260)

    def _pixmap_from_png(self, png_bytes: bytes | None) -> QPixmap | None:
        if not png_bytes:
            return None
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes, "PNG")
        return pixmap

    def _set_placeholder(self, label: QLabel, text: str, min_height: int):
        label.setPixmap(QPixmap())
        label.setText(text)
        label.setMinimumSize(0, min_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        self._refresh_label_pixmap(self.reactions_label, self._reactions_pixmap, minimum_height=220)
        self._refresh_label_pixmap(self.map_label, self._map_pixmap, minimum_height=260)

    def _refresh_label_pixmap(self, label: QLabel, pixmap: QPixmap | None, minimum_height: int):
        if not pixmap or pixmap.isNull():
            return

        show_pixmap_at_natural_size(label, pixmap, minimum_height)


class InputPanel(QWidget):
    def __init__(self, on_predict_callback, on_smiles_ready, on_error, on_clear_callback):
        super().__init__()
        self.on_predict_callback = on_predict_callback
        self.on_smiles_ready = on_smiles_ready
        self.on_error = on_error
        self.on_clear_callback = on_clear_callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        frame = QFrame()
        frame.setStyleSheet(
            """
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            QLabel { color: #f5f7fa; }
            QLineEdit {
                background-color: #1a2030;
                color: #f5f7fa;
                border: 1px solid #3a4151;
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
            }
            """
        )

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        text_label = QLabel("Codigo InChI o SMILES")
        text_label.setStyleSheet("font-size: 14px; font-weight: 700;")

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Pega aca un InChI o un SMILES...")
        self.input_line.returnPressed.connect(self.predict_from_text)

        text_btn_row = QHBoxLayout()

        text_predict_btn = QPushButton("Predecir")
        text_predict_btn.clicked.connect(self.predict_from_text)

        text_clear_btn = QPushButton("Limpiar")
        text_clear_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #30363d;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #3d444d; }
            """
        )
        text_clear_btn.clicked.connect(self.clear_text)

        text_btn_row.addWidget(text_predict_btn)
        text_btn_row.addWidget(text_clear_btn)
        text_btn_row.addStretch()

        editor_label = QLabel("Editor molecular")
        editor_label.setStyleSheet("font-size: 14px; font-weight: 700;")

        self.web = QWebEngineView()
        self.web.setZoomFactor(1.0)
        self.web.setFixedHeight(EDITOR_WEB_HEIGHT)
        self.web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        settings = self.web.settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.load_editor()

        editor_btn_row = QHBoxLayout()

        editor_predict_btn = QPushButton("Predecir desde editor")
        editor_predict_btn.clicked.connect(self.predict_from_editor)

        editor_clear_btn = QPushButton("Limpiar editor")
        editor_clear_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #30363d;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #3d444d; }
            """
        )
        editor_clear_btn.clicked.connect(self.clear_editor)

        editor_btn_row.addWidget(editor_predict_btn)
        editor_btn_row.addWidget(editor_clear_btn)
        editor_btn_row.addStretch()

        outer.addWidget(text_label)
        outer.addWidget(self.input_line)
        outer.addLayout(text_btn_row)
        outer.addSpacing(6)
        outer.addWidget(editor_label)
        outer.addWidget(self.web)
        outer.addLayout(editor_btn_row)
        layout.addWidget(frame)

    def predict_from_text(self):
        self.on_predict_callback(self.input_line.text().strip())

    def clear_text(self):
        self.input_line.clear()
        self.on_clear_callback()

    def load_editor(self):
        if not EDITOR_HTML_PATH.exists():
            self.web.setHtml(
                "<html><body style='background:#11161f;color:white;font-family:Arial'>"
                "<h3>No encontre jsme_editor_embed_fixed.html</h3>"
                "<p>Debe estar en la misma carpeta que la app.</p>"
                "</body></html>"
            )
            return

        html = EDITOR_HTML_PATH.read_text(encoding="utf-8")
        base_url = QUrl.fromLocalFile(str(APP_DIR) + "/")
        self.web.setHtml(html, base_url)

    def predict_from_editor(self):
        js = "window.getCurrentSmiles ? window.getCurrentSmiles() : '';"
        self.web.page().runJavaScript(js, self._handle_smiles)

    def _handle_smiles(self, smiles):
        smiles = (smiles or "").strip()
        if not smiles:
            self.on_error("El editor no devolvio un SMILES valido todavia.")
            return
        self.on_smiles_ready(smiles)

    def clear_editor(self):
        self.web.page().runJavaScript("if(window.clearEditor){window.clearEditor();}")
        self.on_clear_callback()


class ChemicalPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prediccion de propiedades quimicas")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1180, 760)

        self.logp_predictor = None
        self.pkasolver_predictor = PkaSolverPredictor(None, "")
        self.pkasolver_request_serial = 0
        self.pkasolver_workers: list[PkaSolverWorker] = []

        self._load_models()
        self._build_ui()

    def _load_models(self):
        errors = []

        try:
            self.logp_predictor = PropertyPredictor(MODEL_LOGP_PATH, NORM_LOGP_PATH, "log P")
        except Exception as exc:
            errors.append(f"log P: {exc}")

        self.pkasolver_predictor = PkaSolverPredictor.discover(APP_DIR)

        if errors:
            QMessageBox.warning(
                self,
                "Modelos faltantes o invalidos",
                "La app inicio, pero hubo problemas al cargar algunos modelos:\n\n"
                + "\n".join(errors)
                + "\n\nAsegurate de tener los archivos en:\n"
                "models/logp/model_logp.keras y models/logp/logp_norm.npz",
            )

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #11161f;
                color: #f5f7fa;
                font-family: Segoe UI, Arial, sans-serif;
            }
            QPushButton {
                background-color: #2f81f7;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #1f6feb; }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background: #11161f;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #1a2030;
                color: #f5f7fa;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected { background: #2f81f7; }
            """
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(scroll)

        central = QWidget()
        scroll.setWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        title = QLabel("Prediccion de propiedades quimicas desde InChI, SMILES o editor molecular")
        title.setStyleSheet("font-size: 26px; font-weight: 800; color: #ffffff;")

        subtitle = QLabel("Ingresa un InChI o un SMILES, o dibuja la molecula en el editor molecular.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 13px; color: #c9d1d9;")

        self.input_panel = InputPanel(
            self.predict_from_text,
            self.predict_from_smiles,
            self.show_error,
            self.reset_outputs,
        )
        self.input_panel.setMinimumHeight(TOP_SECTION_HEIGHT)

        content = QGridLayout()
        content.setHorizontalSpacing(16)
        content.setVerticalSpacing(16)

        self.viewer = MoleculeViewer()
        self.results = ResultsPanel(self.logp_predictor)
        self.pka_images = PkaImagesPanel()
        self.results.property_changed.connect(self._handle_property_changed)

        content.addWidget(self.viewer, 0, 0)
        content.addWidget(self.results, 0, 1)
        content.setRowStretch(0, 1)
        content.setColumnStretch(0, 3)
        content.setColumnStretch(1, 2)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(self.input_panel)
        root.addLayout(content, stretch=1)
        root.addWidget(self.pka_images)
        self._handle_property_changed(self.results.selected_property())

    def _handle_property_changed(self, property_key: str):
        self.pka_images.setVisible(property_key == "pka")

    def reset_outputs(self):
        self.cancel_pkasolver_prediction()
        self.viewer.clear()
        self.results.reset()
        self.pka_images.clear()

    def show_error(self, message: str):
        self.reset_outputs()
        self.results.status_label.setText(f"Estado: error. {message}")
        QMessageBox.critical(self, "Error de prediccion", message)

    def update_outputs(self, mol, source_kind: str):
        self.cancel_pkasolver_prediction()
        smiles = mol_to_smiles(mol)
        self.viewer.update_for_molecule(mol)
        self.results.update_for_molecule(mol, source_kind)
        self.pka_images.clear()
        self.start_pkasolver_prediction(smiles)

    def predict_from_text(self, raw_text: str):
        try:
            mol, source_kind = parse_structure(raw_text)
            self.update_outputs(mol, source_kind)
        except Exception as exc:
            self.show_error(str(exc))

    def predict_from_smiles(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("El editor devolvio un SMILES invalido.")
            self.update_outputs(mol, "editor molecular")
        except Exception as exc:
            self.show_error(str(exc))

    def start_pkasolver_prediction(self, smiles: str):
        if not self.pkasolver_predictor.is_available():
            self.results.set_pka_error(self.pkasolver_predictor.error_message)
            self.pka_images.set_error(self.pkasolver_predictor.error_message)
            return

        self.pkasolver_request_serial += 1
        request_id = self.pkasolver_request_serial
        worker = PkaSolverWorker(request_id, smiles, self.pkasolver_predictor)
        worker.succeeded.connect(self._handle_pkasolver_result)
        worker.failed.connect(self._handle_pkasolver_error)
        worker.finished.connect(lambda worker=worker: self._cleanup_pkasolver_worker(worker))

        self.pkasolver_workers.append(worker)
        self.results.set_pka_pending()
        self.pka_images.set_pending()
        worker.start()

    def cancel_pkasolver_prediction(self):
        self.pkasolver_request_serial += 1
        for worker in list(self.pkasolver_workers):
            if worker.isRunning():
                worker.requestInterruption()

    def _cleanup_pkasolver_worker(self, worker: PkaSolverWorker):
        if worker in self.pkasolver_workers:
            self.pkasolver_workers.remove(worker)
        worker.deleteLater()

    def _handle_pkasolver_result(self, request_id: int, prediction: PkaSolverPrediction):
        if request_id != self.pkasolver_request_serial:
            return
        self.results.set_pka_result(prediction)
        self.pka_images.set_result(prediction)

    def _handle_pkasolver_error(self, request_id: int, message: str):
        if request_id != self.pkasolver_request_serial:
            return
        self.results.set_pka_error(message)
        self.pka_images.set_error(message)

    def closeEvent(self, event):
        self.cancel_pkasolver_prediction()
        for worker in list(self.pkasolver_workers):
            worker.wait(1000)
        super().closeEvent(event)


def run_pkasolver_smoke_test() -> int:
    log_path = Path(sys.executable).with_suffix(".smoke.log") if getattr(sys, "frozen", False) else SOURCE_DIR / "pkasolver_smoke_test.log"
    log_lines = []

    def log(message: str):
        log_lines.append(message)
        print(message)

    try:
        log(f"Executable: {sys.executable}")
        log(f"APP_DIR: {APP_DIR}")
        predictor = PkaSolverPredictor.discover(APP_DIR)
        if not predictor.is_available():
            log(predictor.error_message)
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            return 2
        prediction = predictor.predict("CC(=O)O")
        if not prediction.states:
            log("pKaSolver no devolvio microestados para el smoke test.")
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            return 3
        log(f"pKaSolver smoke test OK: {len(prediction.states)} microestado(s).")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return 0
    except Exception as exc:
        log(f"pKaSolver smoke test fallo: {exc}")
        log(traceback.format_exc())
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return 1


def main():
    if PKASOLVER_SMOKE_TEST_ARG in sys.argv:
        raise SystemExit(run_pkasolver_smoke_test())

    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except Exception:
            pass

    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = ChemicalPredictorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
