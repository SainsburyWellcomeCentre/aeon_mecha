"""Custom spikeinterface_gui view: multi-select tagging for the unit(s) currently
selected in unitlist - same "selected" concept the built-in quality shortcuts
(g/m/n/c) use, so it feels like one consistent workflow rather than two.

Stored as a single non-exclusive "tags" label category holding a *list* of tags
per unit (bypassing Controller.set_label_to_unit/get_unit_label, which only ever
store/read a single value - see controller.py set_label_to_unit). The Curation
pydantic model does not restrict list length or membership for non-exclusive
categories, only exclusive ones (max 1 value), so this is safe to save/load/export.

See: https://spikeinterface-gui.readthedocs.io/en/latest/custom_views.html
"""

from spikeinterface_gui.view_base import ViewBase

TAGS_CATEGORY = "tags"

# plain-letter shortcuts not already used elsewhere in the GUI (existing: space, c, g, m, n).
# Keys are this view's own choice; values must match the CurationTag lookup - that's the list
# spike_sorting_curation.py reads back off the analyzer when writing SortedSpikes.UnitTag, so a
# mismatch would mean a tag is selectable in the GUI but silently never reaches the database.
# launch_si_gui.py validates this against CurationTag at launch (needs the database).
TAG_SHORTCUTS = {
    "w": "irregular waveform",
    "d": "amplitude drift",
    "b": "bimodal amplitude",
    "t": "intermittent",
    "r": "refractory violations",
    "f": "flag",
}


class MultiTagView(ViewBase):
    id = "multitag"
    _supported_backend = ["qt"]
    _gui_help_txt = (
        "Multi-select feature tags for the unit(s) selected in unitlist (same "
        "selection used by the quality shortcuts g/m/n/c). Click a checkbox or "
        "press its shortcut key to toggle a tag; " + ", ".join(f"{k}={v}" for k, v in TAG_SHORTCUTS.items())
    )
    _settings = None

    def _qt_make_layout(self):
        from spikeinterface_gui.myqt import QT

        self.layout = QT.QVBoxLayout()
        self.qt_widget.setLayout(self.layout)

        self.unit_label = QT.QLabel("Select unit(s) in unitlist")
        self.layout.addWidget(self.unit_label)

        self.checkboxes = {}
        self._shortcuts = []
        for key, tag in TAG_SHORTCUTS.items():
            cb = QT.QCheckBox(f"[{key}]  {tag}")
            cb.stateChanged.connect(lambda _checked, t=tag: self._on_checkbox_toggled(t))
            self.layout.addWidget(cb)
            self.checkboxes[tag] = cb

            shortcut = QT.QShortcut(self.qt_widget)
            shortcut.setKey(QT.QKeySequence(key))
            shortcut.activated.connect(lambda t=tag: self._on_shortcut_toggled(t))
            self._shortcuts.append(shortcut)

        self.layout.addStretch()
        self._connected_to_selection = False

    def _get_unitlist_view(self):
        for view in self.controller.views:
            if view.id == "unitlist":
                return view
        return None

    def _ensure_connected_to_selection(self):
        # unitlist's row-selection has no notify_* broadcast (unlike visibility/curation),
        # so we hook its table's Qt signal directly, once, the first time it's available.
        if self._connected_to_selection:
            return
        unitlist_view = self._get_unitlist_view()
        if unitlist_view is None:
            return
        unitlist_view.table.itemSelectionChanged.connect(self.refresh)
        self._connected_to_selection = True

    def _selected_unit_ids(self):
        unitlist_view = self._get_unitlist_view()
        if unitlist_view is None:
            return []
        return unitlist_view.get_selected_unit_ids()

    def _get_unit_features(self, unit_id):
        for lbl in self.controller.curation_data["manual_labels"]:
            if lbl["unit_id"] == unit_id:
                return list(lbl.get("labels", {}).get(TAGS_CATEGORY, []))
        return []

    def _set_unit_features(self, unit_id, features):
        manual_labels = self.controller.curation_data["manual_labels"]
        for lbl in manual_labels:
            if lbl["unit_id"] == unit_id:
                lbl.setdefault("labels", {})[TAGS_CATEGORY] = features
                return
        manual_labels.append({"unit_id": unit_id, "labels": {TAGS_CATEGORY: features}})

    def _toggle_tag(self, unit_id, tag, add):
        features = self._get_unit_features(unit_id)
        has_tag = tag in features
        if add and not has_tag:
            features.append(tag)
            self._set_unit_features(unit_id, features)
        elif not add and has_tag:
            features.remove(tag)
            self._set_unit_features(unit_id, features)

    def _on_shortcut_toggled(self, tag):
        unit_ids = self._selected_unit_ids()
        if not unit_ids:
            return
        # toggle relative to the first selected unit's current state, applied to all
        add = tag not in self._get_unit_features(unit_ids[0])
        for unit_id in unit_ids:
            self._toggle_tag(unit_id, tag, add)
        self.notify_manual_curation_updated()
        self.refresh()

    def _on_checkbox_toggled(self, tag):
        unit_ids = self._selected_unit_ids()
        if not unit_ids:
            return
        add = self.checkboxes[tag].isChecked()
        for unit_id in unit_ids:
            self._toggle_tag(unit_id, tag, add)
        self.notify_manual_curation_updated()

    def _qt_refresh(self):
        self._ensure_connected_to_selection()
        unit_ids = self._selected_unit_ids()
        if not unit_ids:
            self.unit_label.setText("Select unit(s) in unitlist")
            for cb in self.checkboxes.values():
                cb.setEnabled(False)
            return

        if len(unit_ids) == 1:
            self.unit_label.setText(f"Tags for unit {unit_ids[0]}")
            features = set(self._get_unit_features(unit_ids[0]))
        else:
            self.unit_label.setText(f"Tags for {len(unit_ids)} selected units (shared tags shown)")
            feature_sets = [set(self._get_unit_features(u)) for u in unit_ids]
            features = set.intersection(*feature_sets) if feature_sets else set()

        for tag, cb in self.checkboxes.items():
            cb.setEnabled(True)
            cb.blockSignals(True)
            cb.setChecked(tag in features)
            cb.blockSignals(False)
