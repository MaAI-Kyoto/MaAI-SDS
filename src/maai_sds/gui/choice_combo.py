"""決まった値から選ぶ欄。設定の読み書きは QLineEdit と同じ text() / setText() / editingFinished で行える。

MaAI のモードや言語は、以前は自由に書ける QLineEdit だった（`vap_mc`・`jp` を手で書く）。
読み書きするコード（gui/config_io.py・app_config_spec.py・controller.py）は QLineEdit の API を
前提にしているので、同じ API を持たせた QComboBox に差し替える。

Web の画面定義（scripts/extract_web_ui_manifest.py）は QComboBox として拾い、選択肢つきの
セレクトになる。設定ファイルに一覧に無い値が入っていた場合は、その値を選択肢に足して保つ
（古い設定や、MaAI の新しいモードを手で書いた場合に値を失わないため）。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QWidget


class ChoiceCombo(QComboBox):
    editingFinished = Signal()

    def __init__(self, choices: list[str], value: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.addItems(choices)
        if value:
            self.setText(value)
        # ユーザーが選んだときだけ「編集が終わった」とみなす（setText では出さない）。
        self.activated.connect(lambda _i: self.editingFinished.emit())

    def text(self) -> str:
        return self.currentText()

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit と同じ名前
        value = str(value or "").strip()
        if not value:
            return
        if self.findText(value) < 0:
            self.addItem(value)
        self.setCurrentText(value)
