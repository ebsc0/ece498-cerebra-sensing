"""UI components - Kivy widgets with real-time graphing and head map."""

from collections import deque

from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from config import (
    ACTIVE_OPTODES,
    HEADMAP_IMAGE_PATH,
    MAX_PLOT_POINTS,
    OPTODE_COLORS,
    OPTODE_POSITIONS,
    TOTAL_OPTODES,
)


class MainScreen(Screen):
    """Main screen with tabbed views, session info, ICH alerts, and numeric readouts."""

    def __init__(self, on_start=None, on_stop=None, **kwargs):
        # Import matplotlib dependencies
        from kivy_garden.matplotlib import FigureCanvasKivyAgg
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # Non-interactive backend
        matplotlib.set_loglevel("warning")

        self._plt = plt
        self._FigureCanvasKivyAgg = FigureCanvasKivyAgg

        super().__init__(**kwargs)

        self.on_start_callback = on_start
        self.on_stop_callback = on_stop

        self.theme = {
            "window_bg": (0.94, 0.96, 0.98, 1.0),
            "card_bg": (0.99, 0.995, 1.0, 1.0),
            "card_border": (0.82, 0.86, 0.90, 1.0),
            "text_primary": (0.10, 0.16, 0.23, 1.0),
            "text_secondary": (0.33, 0.40, 0.48, 1.0),
            "accent": (0.08, 0.45, 0.80, 1.0),
            "start_btn": (0.12, 0.53, 0.80, 1.0),
            "stop_btn": (0.78, 0.28, 0.24, 1.0),
            "status_idle_bg": (0.72, 0.76, 0.80, 1.0),
            "status_idle_border": (0.56, 0.60, 0.65, 1.0),
            "status_monitor_bg": (0.30, 0.62, 0.44, 1.0),
            "status_monitor_border": (0.23, 0.50, 0.34, 1.0),
            "status_warn_bg": (0.84, 0.34, 0.28, 1.0),
            "status_warn_border": (0.65, 0.24, 0.20, 1.0),
            "figure_bg": (0.98, 0.99, 1.0, 1.0),
            "axis_text": (0.20, 0.27, 0.34, 1.0),
            "axis_border": (0.73, 0.78, 0.84, 1.0),
            "grid": (0.86, 0.89, 0.93, 1.0),
        }
        Window.clearcolor = self.theme["window_bg"]

        # Graph data: {optode_id: [values...]}
        self.hbo_data = {}
        self.hbr_data = {}
        self.max_points = MAX_PLOT_POINTS

        # Current values for numeric readouts and head map
        self.current_hbo = {}  # {optode_id: value}
        self.current_hbr = {}  # {optode_id: value}

        # ICH flags for head map
        self.ich_flags = {}  # {optode_id: bool}

        # Session tracking
        self.session_id = None
        self.captured_count = 0
        self.processed_count = 0

        # Event log buffering
        self._log_lines = deque(maxlen=1000)

        self._build_ui()

    # -------------------------------------------------------------------------
    # Styling Helpers
    # -------------------------------------------------------------------------

    def _bind_label_text_size(self, label: Label):
        label.bind(size=lambda i, v: setattr(i, "text_size", v))

    def _decorate_root(self, widget: BoxLayout):
        with widget.canvas.before:
            self._root_bg_color = Color(*self.theme["window_bg"])
            self._root_bg_rect = Rectangle(pos=widget.pos, size=widget.size)
        widget.bind(pos=self._update_root_rect, size=self._update_root_rect)

    def _update_root_rect(self, widget, _value):
        self._root_bg_rect.pos = widget.pos
        self._root_bg_rect.size = widget.size

    def _decorate_card(
        self,
        widget,
        bg_rgba=None,
        border_rgba=None,
        radius=12,
    ):
        if bg_rgba is None:
            bg_rgba = self.theme["card_bg"]
        if border_rgba is None:
            border_rgba = self.theme["card_border"]

        with widget.canvas.before:
            bg_color = Color(*bg_rgba)
            bg_rect = RoundedRectangle(
                pos=widget.pos,
                size=widget.size,
                radius=[radius, radius, radius, radius],
            )
            border_color = Color(*border_rgba)
            border_line = Line(
                rounded_rectangle=(widget.x, widget.y, widget.width, widget.height, radius),
                width=1.2,
            )

        widget._card_bg_color = bg_color
        widget._card_bg_rect = bg_rect
        widget._card_border_color = border_color
        widget._card_border_line = border_line
        widget._card_radius = radius
        widget.bind(pos=self._update_card_rect, size=self._update_card_rect)
        return bg_color, border_color

    def _update_card_rect(self, widget, _value):
        if not hasattr(widget, "_card_bg_rect"):
            return
        widget._card_bg_rect.pos = widget.pos
        widget._card_bg_rect.size = widget.size
        widget._card_border_line.rounded_rectangle = (
            widget.x,
            widget.y,
            widget.width,
            widget.height,
            widget._card_radius,
        )

    def _set_button_style(self, button: Button, bg_rgba):
        button.background_normal = ""
        button.background_down = ""
        button.background_color = bg_rgba
        button.color = (1, 1, 1, 1)
        button.bold = True

    def _set_alert_panel_style(self, bg_rgba, border_rgba):
        self._alert_bg_color.rgba = bg_rgba
        self._alert_border_color.rgba = border_rgba

    # -------------------------------------------------------------------------
    # Layout Builders
    # -------------------------------------------------------------------------

    def _build_ui(self):
        root = BoxLayout(orientation="vertical", padding=12, spacing=10)
        self._decorate_root(root)

        root.add_widget(self._build_top_bar())
        root.add_widget(self._build_alert_panel())
        root.add_widget(self._build_tabbed_panel())
        root.add_widget(self._build_bottom_panel())

        self.add_widget(root)

        self._init_graph_plot()
        self._init_head_map()
        self._refresh_readouts()

    def _build_top_bar(self):
        top_card = BoxLayout(
            size_hint=(1, 0.12),
            spacing=14,
            padding=(12, 10),
        )
        self._decorate_card(top_card)

        # Left: title
        title_box = BoxLayout(orientation="vertical", size_hint=(0.38, 1), spacing=2)
        title = Label(
            text="Cerebra Hemoglobin Monitor",
            halign="left",
            valign="middle",
            bold=True,
            color=self.theme["text_primary"],
            font_size="20sp",
        )
        subtitle = Label(
            text="Real-time cerebral perfusion tracking",
            halign="left",
            valign="middle",
            color=self.theme["text_secondary"],
            font_size="13sp",
        )
        self._bind_label_text_size(title)
        self._bind_label_text_size(subtitle)
        title_box.add_widget(title)
        title_box.add_widget(subtitle)

        # Middle: session info
        session_box = BoxLayout(orientation="vertical", size_hint=(0.38, 1), spacing=2)
        self.session_info_label = Label(
            text="SESSION: -- | --:--:--",
            halign="left",
            valign="middle",
            color=self.theme["text_primary"],
            font_size="14sp",
            bold=True,
        )
        self.stream_state_label = Label(
            text="Status: Idle | Captured: 0 | Processed: 0",
            halign="left",
            valign="middle",
            color=self.theme["text_secondary"],
            font_size="13sp",
        )
        self._bind_label_text_size(self.session_info_label)
        self._bind_label_text_size(self.stream_state_label)
        session_box.add_widget(self.session_info_label)
        session_box.add_widget(self.stream_state_label)

        # Right: controls
        controls = BoxLayout(size_hint=(0.24, 1), spacing=8)
        self.start_btn = Button(text="Start", size_hint=(0.5, 1))
        self.stop_btn = Button(text="Stop", size_hint=(0.5, 1), disabled=True)
        self._set_button_style(self.start_btn, self.theme["start_btn"])
        self._set_button_style(self.stop_btn, self.theme["stop_btn"])
        self.start_btn.bind(on_press=self._on_start_pressed)
        self.stop_btn.bind(on_press=self._on_stop_pressed)
        controls.add_widget(self.start_btn)
        controls.add_widget(self.stop_btn)

        top_card.add_widget(title_box)
        top_card.add_widget(session_box)
        top_card.add_widget(controls)
        return top_card

    def _build_alert_panel(self):
        alert_card = BoxLayout(
            orientation="vertical",
            size_hint=(1, 0.10),
            spacing=2,
            padding=(12, 8),
        )
        self._alert_bg_color, self._alert_border_color = self._decorate_card(
            alert_card,
            bg_rgba=self.theme["status_idle_bg"],
            border_rgba=self.theme["status_idle_border"],
            radius=10,
        )

        self.ich_status_label = Label(
            text="ICH Status: Idle",
            halign="left",
            valign="middle",
            color=(1, 1, 1, 1),
            bold=True,
            font_size="17sp",
        )
        self.ich_detail_label = Label(
            text="No active monitoring session",
            halign="left",
            valign="middle",
            color=(0.95, 0.97, 1.0, 1.0),
            font_size="12sp",
        )
        self._bind_label_text_size(self.ich_status_label)
        self._bind_label_text_size(self.ich_detail_label)

        alert_card.add_widget(self.ich_status_label)
        alert_card.add_widget(self.ich_detail_label)
        return alert_card

    def _build_tabbed_panel(self):
        self.tabbed_panel = TabbedPanel(
            size_hint=(1, 0.50),
            do_default_tab=False,
            tab_height=34,
            tab_width=180,
            background_color=self.theme["card_bg"],
        )

        graph_tab = TabbedPanelItem(text="Trend Graph")
        self.graph_fig, self.graph_ax = self._plt.subplots(figsize=(8, 4))
        self.graph_fig.tight_layout(pad=2.0)
        self.graph_widget = self._FigureCanvasKivyAgg(self.graph_fig)
        graph_tab.add_widget(self.graph_widget)

        headmap_tab = TabbedPanelItem(text="Optode Map")
        self.headmap_fig, self.headmap_ax = self._plt.subplots(figsize=(5, 5))
        self.headmap_fig.tight_layout(pad=1.0)
        self.headmap_widget = self._FigureCanvasKivyAgg(self.headmap_fig)
        headmap_tab.add_widget(self.headmap_widget)

        self.tabbed_panel.add_widget(graph_tab)
        self.tabbed_panel.add_widget(headmap_tab)
        self.tabbed_panel.default_tab = graph_tab
        return self.tabbed_panel

    def _build_bottom_panel(self):
        bottom = BoxLayout(size_hint=(1, 0.28), spacing=10)

        readouts_card = BoxLayout(
            orientation="vertical",
            size_hint=(0.40, 1),
            spacing=6,
            padding=(10, 8),
        )
        self._decorate_card(readouts_card)
        readouts_header = Label(
            text="Optode Readout",
            size_hint=(1, 0.16),
            halign="left",
            valign="middle",
            bold=True,
            color=self.theme["text_primary"],
            font_size="15sp",
        )
        self._bind_label_text_size(readouts_header)
        readouts_scroll = ScrollView(size_hint=(1, 0.84))
        self.readouts_label = Label(
            text="Waiting for processed data...",
            size_hint_y=None,
            halign="left",
            valign="top",
            markup=True,
            color=self.theme["text_primary"],
            font_size="13sp",
        )
        self.readouts_label.bind(texture_size=self._update_readouts_height)
        readouts_scroll.add_widget(self.readouts_label)
        self._readouts_scroll = readouts_scroll
        readouts_card.add_widget(readouts_header)
        readouts_card.add_widget(readouts_scroll)

        log_card = BoxLayout(
            orientation="vertical",
            size_hint=(0.60, 1),
            spacing=6,
            padding=(10, 8),
        )
        self._decorate_card(log_card)
        log_header = Label(
            text="System Timeline",
            size_hint=(1, 0.16),
            halign="left",
            valign="middle",
            bold=True,
            color=self.theme["text_primary"],
            font_size="15sp",
        )
        self._bind_label_text_size(log_header)
        log_scroll = ScrollView(size_hint=(1, 0.84))
        self.log_label = Label(
            text="",
            size_hint_y=None,
            halign="left",
            valign="top",
            markup=False,
            color=self.theme["text_secondary"],
            font_size="12sp",
        )
        self.log_label.bind(texture_size=self._update_log_height)
        log_scroll.add_widget(self.log_label)
        self._log_scroll = log_scroll
        log_card.add_widget(log_header)
        log_card.add_widget(log_scroll)

        bottom.add_widget(readouts_card)
        bottom.add_widget(log_card)
        return bottom

    def _update_readouts_height(self, instance, value):
        instance.height = value[1]
        instance.text_size = (self._readouts_scroll.width - 10, None)

    def _update_log_height(self, instance, value):
        instance.height = value[1]
        instance.text_size = (self._log_scroll.width - 10, None)

    # -------------------------------------------------------------------------
    # Graph Tab Methods
    # -------------------------------------------------------------------------

    def _style_graph_axes(self):
        ax = self.graph_ax
        ax.set_facecolor(self.theme["figure_bg"])
        ax.tick_params(colors=self.theme["axis_text"], labelsize=8)
        ax.set_xlabel("Samples (Most Recent)", color=self.theme["axis_text"])
        ax.set_ylabel("Delta Hb (a.u.)", color=self.theme["axis_text"])
        ax.set_title("Hemoglobin Trend (Long - Short)", color=self.theme["text_primary"], fontsize=11)
        ax.grid(True, color=self.theme["grid"], linewidth=0.8, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color(self.theme["axis_border"])

    def _init_graph_plot(self):
        self.graph_fig.patch.set_facecolor(self.theme["figure_bg"])
        self._style_graph_axes()
        self.graph_widget.draw()

    def update_graph(self, preprocessed_data: dict):
        """Update graph with new preprocessed data from one frame."""
        for optode_id, result in preprocessed_data.items():
            hbo = result.hbo_long - result.hbo_short
            hbr = result.hbr_long - result.hbr_short

            self.current_hbo[optode_id] = hbo
            self.current_hbr[optode_id] = hbr

            self.hbo_data.setdefault(optode_id, []).append(hbo)
            self.hbr_data.setdefault(optode_id, []).append(hbr)

            if len(self.hbo_data[optode_id]) > self.max_points:
                self.hbo_data[optode_id] = self.hbo_data[optode_id][-self.max_points:]
            if len(self.hbr_data[optode_id]) > self.max_points:
                self.hbr_data[optode_id] = self.hbr_data[optode_id][-self.max_points:]

        self._refresh_graph()
        self._refresh_readouts()

    def _refresh_graph(self):
        self.graph_ax.clear()
        self._style_graph_axes()

        for optode_id in sorted(self.hbo_data.keys()):
            color = OPTODE_COLORS[optode_id % len(OPTODE_COLORS)]
            hbo_values = self.hbo_data.get(optode_id, [])
            hbr_values = self.hbr_data.get(optode_id, [])

            if hbo_values:
                self.graph_ax.plot(
                    hbo_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    label=f"Optode {optode_id} HbO",
                )
            if hbr_values:
                self.graph_ax.plot(
                    hbr_values,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.95,
                    label=f"Optode {optode_id} HbR",
                )

        if self.hbo_data or self.hbr_data:
            legend = self.graph_ax.legend(
                loc="upper right",
                fontsize="x-small",
                ncol=2,
                frameon=True,
                facecolor=self.theme["figure_bg"],
            )
            legend.get_frame().set_edgecolor(self.theme["axis_border"])

        self.graph_widget.draw()

    # -------------------------------------------------------------------------
    # Head Map Tab Methods
    # -------------------------------------------------------------------------

    def _init_head_map(self):
        """Initialize head map with optode positions over background image."""
        from matplotlib.patches import Circle
        import matplotlib.image as mpimg
        import os

        self.headmap_fig.patch.set_facecolor(self.theme["figure_bg"])
        self.headmap_ax.clear()
        if hasattr(self, "headmap_colorbar") and self.headmap_colorbar is not None:
            self.headmap_colorbar.remove()
            self.headmap_colorbar = None

        if HEADMAP_IMAGE_PATH and os.path.exists(HEADMAP_IMAGE_PATH):
            try:
                img = mpimg.imread(HEADMAP_IMAGE_PATH)
                self.headmap_ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto", zorder=0)
            except Exception as e:
                print(f"Warning: Could not load headmap image: {e}")

        self.headmap_ax.set_xlim(0, 1)
        self.headmap_ax.set_ylim(0, 1)
        self.headmap_ax.set_aspect("equal")
        self.headmap_ax.axis("off")
        self.headmap_ax.set_title("Optode Layout (Delta HbO)", fontsize=11, color=self.theme["text_primary"])

        self.optode_circles = {}
        self.optode_labels = {}

        for optode_id in range(TOTAL_OPTODES):
            x, y = OPTODE_POSITIONS.get(optode_id, (0.5, 0.5))
            if optode_id in ACTIVE_OPTODES:
                circle = Circle(
                    (x, y),
                    0.035,
                    facecolor=(0.72, 0.74, 0.77, 0.90),
                    edgecolor=(0.20, 0.24, 0.30, 1.0),
                    linewidth=1.2,
                    zorder=2,
                )
            else:
                circle = Circle(
                    (x, y),
                    0.035,
                    fill=False,
                    edgecolor=(0.58, 0.61, 0.65, 1.0),
                    linestyle="--",
                    linewidth=1.0,
                    zorder=2,
                )

            self.headmap_ax.add_patch(circle)
            self.optode_circles[optode_id] = circle

            label = self.headmap_ax.text(
                x,
                y,
                str(optode_id),
                ha="center",
                va="center",
                fontsize=7,
                color=(0.12, 0.16, 0.22, 1.0) if optode_id in ACTIVE_OPTODES else (0.46, 0.49, 0.54, 1.0),
                zorder=3,
            )
            self.optode_labels[optode_id] = label

        self._setup_colorbar()
        self.headmap_widget.draw()

    def _setup_colorbar(self):
        import matplotlib.colors as mcolors
        from matplotlib.cm import ScalarMappable, get_cmap

        self.headmap_cmap = get_cmap("RdYlBu_r")
        self.headmap_norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)
        sm = ScalarMappable(cmap=self.headmap_cmap, norm=self.headmap_norm)
        sm.set_array([])

        self.headmap_colorbar = self.headmap_fig.colorbar(
            sm,
            ax=self.headmap_ax,
            orientation="horizontal",
            fraction=0.05,
            pad=0.03,
            label="Delta HbO (a.u.)",
        )
        self.headmap_colorbar.ax.tick_params(labelsize=8, colors=self.theme["axis_text"])
        self.headmap_colorbar.outline.set_edgecolor(self.theme["axis_border"])
        self.headmap_colorbar.ax.xaxis.label.set_color(self.theme["axis_text"])

    def _label_color_for_fill(self, rgba):
        r, g, b = rgba[:3]
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        if luminance > 0.58:
            return (0.10, 0.14, 0.20, 1.0)
        return (1.0, 1.0, 1.0, 1.0)

    def _refresh_head_map(self):
        for optode_id in range(TOTAL_OPTODES):
            circle = self.optode_circles.get(optode_id)
            label = self.optode_labels.get(optode_id)
            if circle is None:
                continue

            if optode_id in ACTIVE_OPTODES:
                hbo_value = self.current_hbo.get(optode_id, 0.0)
                fill = self.headmap_cmap(self.headmap_norm(hbo_value))
                circle.set_facecolor(fill)
                circle.set_alpha(0.88)

                if self.ich_flags.get(optode_id, False):
                    circle.set_edgecolor((0.83, 0.17, 0.14, 1.0))
                    circle.set_linewidth(2.8)
                else:
                    circle.set_edgecolor((0.20, 0.24, 0.30, 1.0))
                    circle.set_linewidth(1.2)

                label.set_color(self._label_color_for_fill(fill))

        self.headmap_widget.draw()

    # -------------------------------------------------------------------------
    # Numeric Readouts
    # -------------------------------------------------------------------------

    def _refresh_readouts(self):
        lines = []
        if not self.current_hbo and not self.current_hbr:
            lines.append("[color=566576]Waiting for processed data...[/color]")
            self.readouts_label.text = "\n".join(lines)
            return

        lines.append("[b]Optode | HbO Delta | HbR Delta | Status[/b]")
        lines.append("")
        for optode_id in sorted(ACTIVE_OPTODES):
            hbo = self.current_hbo.get(optode_id, 0.0)
            hbr = self.current_hbr.get(optode_id, 0.0)
            flagged = self.ich_flags.get(optode_id, False)

            status_text = "ALERT" if flagged else "NORMAL"
            status_color = "cc2b23" if flagged else "287a46"
            lines.append(
                f"Optode {optode_id:02d} | HbO {hbo:+.4f} | HbR {hbr:+.4f} | "
                f"[color={status_color}][b]{status_text}[/b][/color]"
            )

        self.readouts_label.text = "\n".join(lines)

    # -------------------------------------------------------------------------
    # ICH Status and Session Info
    # -------------------------------------------------------------------------

    def update_ich_status(self, flags: dict, flag_counts: dict = None):
        """Update ICH status display."""
        self.ich_flags = flags or {}
        any_flagged = any(self.ich_flags.values())

        if not self.ich_flags:
            if self.session_id:
                self.ich_status_label.text = "ICH Status: Baseline / Warmup"
                self.ich_detail_label.text = "Collecting baseline before reliable detection."
                self._set_alert_panel_style(
                    self.theme["status_idle_bg"],
                    self.theme["status_idle_border"],
                )
            else:
                self.ich_status_label.text = "ICH Status: Idle"
                self.ich_detail_label.text = "No active monitoring session."
                self._set_alert_panel_style(
                    self.theme["status_idle_bg"],
                    self.theme["status_idle_border"],
                )
        elif any_flagged:
            flagged_ids = [k for k, v in self.ich_flags.items() if v]
            flagged_text = ", ".join(str(i) for i in flagged_ids)
            self.ich_status_label.text = f"ICH Status: Alert (Optodes {flagged_text})"
            if flag_counts:
                details = [f"{i}:{flag_counts.get(i, 0)} criteria" for i in flagged_ids]
                self.ich_detail_label.text = "Escalation criteria met: " + ", ".join(details)
            else:
                self.ich_detail_label.text = "Escalation criteria met. Review patient condition."
            self._set_alert_panel_style(
                self.theme["status_warn_bg"],
                self.theme["status_warn_border"],
            )
        else:
            self.ich_status_label.text = "ICH Status: Monitoring"
            self.ich_detail_label.text = "No active hemorrhage indicators detected."
            self._set_alert_panel_style(
                self.theme["status_monitor_bg"],
                self.theme["status_monitor_border"],
            )

        self._refresh_head_map()
        self._refresh_readouts()

    def update_session_info(
        self,
        session_id,
        elapsed_str: str,
        captured_count: int,
        processed_count: int,
    ):
        """Update session info display."""
        self.session_id = session_id
        self.captured_count = captured_count
        self.processed_count = processed_count

        if session_id:
            self.session_info_label.text = f"SESSION: {session_id} | {elapsed_str}"
            stream_state = "Baseline Warmup" if captured_count > 0 and processed_count == 0 else "Live"
            self.stream_state_label.text = (
                f"Status: {stream_state} | Captured: {captured_count} | Processed: {processed_count}"
            )
        else:
            self.session_info_label.text = "SESSION: -- | --:--:--"
            self.stream_state_label.text = "Status: Idle | Captured: 0 | Processed: 0"

    # -------------------------------------------------------------------------
    # Button Handlers
    # -------------------------------------------------------------------------

    def _on_start_pressed(self, _instance):
        self.start_btn.disabled = True
        self.stop_btn.disabled = False

        self.hbo_data.clear()
        self.hbr_data.clear()
        self.current_hbo.clear()
        self.current_hbr.clear()
        self.ich_flags.clear()

        self._init_graph_plot()
        self._init_head_map()
        self._refresh_readouts()

        self.ich_status_label.text = "ICH Status: Monitoring"
        self.ich_detail_label.text = "Acquisition started. Monitoring incoming data."
        self._set_alert_panel_style(
            self.theme["status_monitor_bg"],
            self.theme["status_monitor_border"],
        )

        if self.on_start_callback:
            self.on_start_callback()

    def _on_stop_pressed(self, _instance):
        self.start_btn.disabled = False
        self.stop_btn.disabled = True

        self.ich_status_label.text = "ICH Status: Idle"
        self.ich_detail_label.text = "Acquisition stopped."
        self._set_alert_panel_style(
            self.theme["status_idle_bg"],
            self.theme["status_idle_border"],
        )

        if self.on_stop_callback:
            self.on_stop_callback()

    # -------------------------------------------------------------------------
    # Event Log
    # -------------------------------------------------------------------------

    def append_log(self, text: str):
        """Append text to event log and auto-scroll to bottom."""
        self._log_lines.extend(text.splitlines(keepends=True))
        self.log_label.text = "".join(self._log_lines)
        self._log_scroll.scroll_y = 0

    def clear_log(self):
        """Clear the event log."""
        self._log_lines.clear()
        self.log_label.text = ""
