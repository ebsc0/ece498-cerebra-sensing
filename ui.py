"""UI components - Kivy widgets with real-time graphing and head map."""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.graphics import Color, Rectangle

from config import MAX_PLOT_POINTS, OPTODE_COLORS, TOTAL_OPTODES, ACTIVE_OPTODES, OPTODE_POSITIONS, HEADMAP_IMAGE_PATH


class MainScreen(Screen):
    """Main screen with tabbed views, session info, ICH alerts, and numeric readouts."""

    def __init__(self, on_start=None, on_stop=None, **kwargs):
        # Import matplotlib dependencies
        from kivy_garden.matplotlib import FigureCanvasKivyAgg
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        matplotlib.set_loglevel("warning")

        self._plt = plt
        self._FigureCanvasKivyAgg = FigureCanvasKivyAgg

        super().__init__(**kwargs)

        self.on_start_callback = on_start
        self.on_stop_callback = on_stop

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
        self.frame_count = 0

        self._build_ui()

    def _build_ui(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=5)

        # ─── Top Bar: Buttons + Session Info ───────────────────────────────────
        root.add_widget(self._build_top_bar())

        # ─── ICH Alert Panel ───────────────────────────────────────────────────
        root.add_widget(self._build_alert_panel())

        # ─── Tabbed Content Area (Graph / Head Map) ────────────────────────────
        root.add_widget(self._build_tabbed_panel())

        # ─── Bottom Section: Numeric Readouts | Event Log ──────────────────────
        root.add_widget(self._build_bottom_panel())

        self.add_widget(root)

        # Initialize plots
        self._init_graph_plot()
        self._init_head_map()

    def _build_top_bar(self):
        """Build top bar with Start/Stop buttons and session info."""
        top_bar = BoxLayout(size_hint=(1, 0.06), spacing=10)

        self.start_btn = Button(text='Start', size_hint=(0.12, 1))
        self.stop_btn = Button(text='Stop', size_hint=(0.12, 1), disabled=True)
        self.start_btn.bind(on_press=self._on_start_pressed)
        self.stop_btn.bind(on_press=self._on_stop_pressed)

        self.session_info_label = Label(
            text='SESSION: -- | --:--:-- | 0 frames',
            size_hint=(0.76, 1),
            halign='right',
            valign='middle'
        )
        self.session_info_label.bind(size=lambda i, v: setattr(i, 'text_size', v))

        top_bar.add_widget(self.start_btn)
        top_bar.add_widget(self.stop_btn)
        top_bar.add_widget(self.session_info_label)

        return top_bar

    def _build_alert_panel(self):
        """Build ICH status alert panel with color-coded background."""
        self.alert_panel = BoxLayout(size_hint=(1, 0.05))

        self.ich_status_label = Label(
            text='ICH STATUS: IDLE',
            size_hint=(1, 1),
            halign='center',
            valign='middle',
            bold=True
        )
        self.ich_status_label.bind(size=lambda i, v: setattr(i, 'text_size', v))

        # Set initial background color (gray for idle)
        with self.alert_panel.canvas.before:
            self._alert_color = Color(0.3, 0.3, 0.3, 1)  # Gray
            self._alert_rect = Rectangle(pos=self.alert_panel.pos, size=self.alert_panel.size)

        self.alert_panel.bind(pos=self._update_alert_rect, size=self._update_alert_rect)
        self.alert_panel.add_widget(self.ich_status_label)

        return self.alert_panel

    def _update_alert_rect(self, *args):
        """Update alert panel background rectangle."""
        self._alert_rect.pos = self.alert_panel.pos
        self._alert_rect.size = self.alert_panel.size

    def _build_tabbed_panel(self):
        """Build tabbed panel with Graph and Head Map tabs."""
        self.tabbed_panel = TabbedPanel(
            size_hint=(1, 0.52),
            do_default_tab=False,
            tab_height=30
        )

        # ─── Graph Tab ─────────────────────────────────────────────────────────
        graph_tab = TabbedPanelItem(text='Graph')
        self.graph_fig, self.graph_ax = self._plt.subplots(figsize=(8, 4))
        self.graph_fig.tight_layout(pad=2.0)
        self.graph_widget = self._FigureCanvasKivyAgg(self.graph_fig)
        graph_tab.add_widget(self.graph_widget)

        # ─── Head Map Tab ──────────────────────────────────────────────────────
        headmap_tab = TabbedPanelItem(text='Head Map')
        self.headmap_fig, self.headmap_ax = self._plt.subplots(figsize=(5, 5))
        self.headmap_fig.tight_layout(pad=1.0)
        self.headmap_widget = self._FigureCanvasKivyAgg(self.headmap_fig)
        headmap_tab.add_widget(self.headmap_widget)

        self.tabbed_panel.add_widget(graph_tab)
        self.tabbed_panel.add_widget(headmap_tab)

        # Set default tab
        self.tabbed_panel.default_tab = graph_tab

        return self.tabbed_panel

    def _build_bottom_panel(self):
        """Build bottom section with numeric readouts and event log."""
        bottom = BoxLayout(size_hint=(1, 0.37), spacing=10)

        # ─── Numeric Readouts (Left) ───────────────────────────────────────────
        readouts_container = BoxLayout(orientation='vertical', size_hint=(0.35, 1))

        readouts_header = Label(
            text='Current Values',
            size_hint=(1, 0.12),
            halign='left',
            valign='middle',
            bold=True
        )
        readouts_header.bind(size=lambda i, v: setattr(i, 'text_size', v))

        readouts_scroll = ScrollView(size_hint=(1, 0.88))
        self.readouts_label = Label(
            text='',
            size_hint_y=None,
            halign='left',
            valign='top',
            markup=True
        )
        self.readouts_label.bind(texture_size=self._update_readouts_height)
        readouts_scroll.add_widget(self.readouts_label)
        self._readouts_scroll = readouts_scroll

        readouts_container.add_widget(readouts_header)
        readouts_container.add_widget(readouts_scroll)

        # ─── Event Log (Right) ─────────────────────────────────────────────────
        log_container = BoxLayout(orientation='vertical', size_hint=(0.65, 1))

        log_header = Label(
            text='Event Log',
            size_hint=(1, 0.12),
            halign='left',
            valign='middle',
            bold=True
        )
        log_header.bind(size=lambda i, v: setattr(i, 'text_size', v))

        log_scroll = ScrollView(size_hint=(1, 0.88))
        self.log_label = Label(
            text='',
            size_hint_y=None,
            halign='left',
            valign='top',
            markup=True
        )
        self.log_label.bind(texture_size=self._update_log_height)
        log_scroll.add_widget(self.log_label)
        self._log_scroll = log_scroll

        log_container.add_widget(log_header)
        log_container.add_widget(log_scroll)

        bottom.add_widget(readouts_container)
        bottom.add_widget(log_container)

        return bottom

    def _update_readouts_height(self, instance, value):
        instance.height = value[1]
        instance.text_size = (self._readouts_scroll.width - 10, None)

    def _update_log_height(self, instance, value):
        instance.height = value[1]
        instance.text_size = (self._log_scroll.width - 10, None)

    # ─────────────────────────────────────────────────────────────────────────────
    # Graph Tab Methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _init_graph_plot(self):
        """Initialize empty graph plot with labels."""
        self.graph_ax.set_xlabel('Sample')
        self.graph_ax.set_ylabel('Delta Hb (a.u.)')
        self.graph_ax.set_title('Hemoglobin Changes (HbO/HbR)')
        self.graph_ax.grid(True, alpha=0.3)
        self.graph_widget.draw()

    def update_graph(self, preprocessed_data: dict):
        """Update graph with new preprocessed data from one frame.

        Args:
            preprocessed_data: {optode_id: PreprocessedResult} from preprocessor.
                Each result has hbo_short, hbo_long, hbr_short, hbr_long.
        """
        for optode_id, result in preprocessed_data.items():
            # Compute difference: long - short
            hbo = result.hbo_long - result.hbo_short
            hbr = result.hbr_long - result.hbr_short

            # Store current values for readouts and head map
            self.current_hbo[optode_id] = hbo
            self.current_hbr[optode_id] = hbr

            # Initialize if new optode
            if optode_id not in self.hbo_data:
                self.hbo_data[optode_id] = []
            if optode_id not in self.hbr_data:
                self.hbr_data[optode_id] = []

            # Append values
            self.hbo_data[optode_id].append(hbo)
            self.hbr_data[optode_id].append(hbr)

            # Trim to max points
            if len(self.hbo_data[optode_id]) > self.max_points:
                self.hbo_data[optode_id] = self.hbo_data[optode_id][-self.max_points:]
            if len(self.hbr_data[optode_id]) > self.max_points:
                self.hbr_data[optode_id] = self.hbr_data[optode_id][-self.max_points:]

        self._refresh_graph()
        self._refresh_head_map()
        self._refresh_readouts()

    def _refresh_graph(self):
        """Redraw the graph with current data."""
        self.graph_ax.clear()
        self.graph_ax.set_xlabel('Sample')
        self.graph_ax.set_ylabel('Delta Hb (a.u.)')
        self.graph_ax.set_title('Hemoglobin Changes (HbO/HbR)')
        self.graph_ax.grid(True, alpha=0.3)

        # Plot each optode's data
        for optode_id in sorted(self.hbo_data.keys()):
            color_idx = optode_id % len(OPTODE_COLORS)
            color = OPTODE_COLORS[color_idx]

            hbo_values = self.hbo_data.get(optode_id, [])
            hbr_values = self.hbr_data.get(optode_id, [])

            if hbo_values:
                self.graph_ax.plot(
                    hbo_values,
                    color=color,
                    linestyle='-',
                    linewidth=1.5,
                    label=f'Optode {optode_id} HbO'
                )
            if hbr_values:
                self.graph_ax.plot(
                    hbr_values,
                    color=color,
                    linestyle='--',
                    linewidth=1.5,
                    label=f'Optode {optode_id} HbR'
                )

        # Add legend if we have data
        if self.hbo_data or self.hbr_data:
            self.graph_ax.legend(loc='upper right', fontsize='x-small', ncol=2)

        self.graph_widget.draw()

    # ─────────────────────────────────────────────────────────────────────────────
    # Head Map Tab Methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _init_head_map(self):
        """Initialize head map with optode positions over background image."""
        from matplotlib.patches import Circle
        import matplotlib.image as mpimg
        import os

        self.headmap_ax.clear()

        # Try to load background image
        image_loaded = False
        if HEADMAP_IMAGE_PATH and os.path.exists(HEADMAP_IMAGE_PATH):
            try:
                img = mpimg.imread(HEADMAP_IMAGE_PATH)
                self.headmap_ax.imshow(img, extent=(0, 1, 0, 1), aspect='auto', zorder=0)
                image_loaded = True
            except Exception as e:
                print(f"Warning: Could not load headmap image: {e}")

        # Set up axes
        self.headmap_ax.set_xlim(0, 1)
        self.headmap_ax.set_ylim(0, 1)
        self.headmap_ax.set_aspect('equal')
        self.headmap_ax.axis('off')
        self.headmap_ax.set_title('Optode Layout (HbO Intensity)', fontsize=10)

        # Store optode circle references for updates
        self.optode_circles = {}
        self.optode_labels = {}

        # Draw optode circles
        for optode_id in range(TOTAL_OPTODES):
            x, y = OPTODE_POSITIONS.get(optode_id, (0.5, 0.5))

            if optode_id in ACTIVE_OPTODES:
                # Active optode - filled circle (will be colored by intensity)
                circle = Circle(
                    (x, y), 0.035,
                    color='gray', alpha=0.7,
                    edgecolor='black', linewidth=1,
                    zorder=2
                )
            else:
                # Inactive optode - gray dashed outline
                circle = Circle(
                    (x, y), 0.035,
                    fill=False,
                    edgecolor='gray', linestyle='--', linewidth=1,
                    zorder=2
                )

            self.headmap_ax.add_patch(circle)
            self.optode_circles[optode_id] = circle

            # Add optode ID label
            label = self.headmap_ax.text(
                x, y, str(optode_id),
                ha='center', va='center', fontsize=7,
                color='white' if optode_id in ACTIVE_OPTODES else 'gray',
                zorder=3
            )
            self.optode_labels[optode_id] = label

        # Add colorbar placeholder
        self._setup_colorbar()

        self.headmap_widget.draw()

    def _setup_colorbar(self):
        """Set up colorbar for head map."""
        import matplotlib.colors as mcolors
        from matplotlib.cm import ScalarMappable, get_cmap

        # Create a colormap (coolwarm: blue=low, red=high)
        self.headmap_cmap = get_cmap('coolwarm')
        self.headmap_norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)

        # Create ScalarMappable for colorbar
        sm = ScalarMappable(cmap=self.headmap_cmap, norm=self.headmap_norm)
        sm.set_array([])

        # Add colorbar
        self.headmap_colorbar = self.headmap_fig.colorbar(
            sm, ax=self.headmap_ax,
            orientation='horizontal',
            fraction=0.05, pad=0.02,
            label='HbO (a.u.)'
        )

    def _refresh_head_map(self):
        """Update head map optode colors based on current HbO values."""
        for optode_id in range(TOTAL_OPTODES):
            circle = self.optode_circles.get(optode_id)
            label = self.optode_labels.get(optode_id)

            if circle is None:
                continue

            if optode_id in ACTIVE_OPTODES:
                # Get current HbO value
                hbo_value = self.current_hbo.get(optode_id, 0)

                # Map to color
                color = self.headmap_cmap(self.headmap_norm(hbo_value))
                circle.set_facecolor(color)
                circle.set_alpha(0.8)

                # Check ICH flag - add red border if flagged
                if self.ich_flags.get(optode_id, False):
                    circle.set_edgecolor('red')
                    circle.set_linewidth(3)
                else:
                    circle.set_edgecolor('black')
                    circle.set_linewidth(1)

                # Update label color for contrast
                label.set_color('white' if hbo_value < 0 else 'black')

        self.headmap_widget.draw()

    # ─────────────────────────────────────────────────────────────────────────────
    # Numeric Readouts
    # ─────────────────────────────────────────────────────────────────────────────

    def _refresh_readouts(self):
        """Update numeric readouts with current values."""
        lines = []
        for optode_id in sorted(ACTIVE_OPTODES):
            hbo = self.current_hbo.get(optode_id, 0)
            hbr = self.current_hbr.get(optode_id, 0)

            # Format with color indication
            hbo_color = '[color=ff0000]' if hbo > 0.3 else '[color=ffffff]'
            hbr_color = '[color=0088ff]' if hbr < -0.3 else '[color=ffffff]'

            lines.append(f"[b]Optode {optode_id}[/b]")
            lines.append(f"  {hbo_color}HbO: {hbo:+.4f}[/color]")
            lines.append(f"  {hbr_color}HbR: {hbr:+.4f}[/color]")
            lines.append("")

        self.readouts_label.text = '\n'.join(lines)

    # ─────────────────────────────────────────────────────────────────────────────
    # ICH Status and Session Info
    # ─────────────────────────────────────────────────────────────────────────────

    def update_ich_status(self, flags: dict, flag_counts: dict = None):
        """Update ICH status display.

        Args:
            flags: {optode_id: bool} - True if ICH detected at optode
            flag_counts: {optode_id: int} - number of detection criteria met (optional)
        """
        self.ich_flags = flags

        # Check if any optode is flagged
        any_flagged = any(flags.values())
        flagged_count = sum(1 for f in flags.values() if f)

        if not flags:
            # No data yet
            self.ich_status_label.text = 'ICH STATUS: IDLE'
            self._alert_color.rgba = (0.3, 0.3, 0.3, 1)  # Gray
        elif any_flagged:
            # ICH detected
            flagged_ids = [str(k) for k, v in flags.items() if v]
            self.ich_status_label.text = f'ICH STATUS: ALERT - Optodes {", ".join(flagged_ids)}'
            self._alert_color.rgba = (0.8, 0.2, 0.2, 1)  # Red
        else:
            # Monitoring, no issues
            self.ich_status_label.text = 'ICH STATUS: MONITORING'
            self._alert_color.rgba = (0.2, 0.6, 0.2, 1)  # Green

        # Refresh head map to show ICH markers
        self._refresh_head_map()

    def update_session_info(self, session_id, elapsed_str: str, frame_count: int):
        """Update session info display.

        Args:
            session_id: Current session ID or None
            elapsed_str: Formatted elapsed time (HH:MM:SS)
            frame_count: Number of frames processed
        """
        self.session_id = session_id
        self.frame_count = frame_count

        if session_id:
            self.session_info_label.text = f'SESSION: {session_id} | {elapsed_str} | {frame_count} frames'
        else:
            self.session_info_label.text = 'SESSION: -- | --:--:-- | 0 frames'

    # ─────────────────────────────────────────────────────────────────────────────
    # Button Handlers
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_start_pressed(self, instance):
        self.start_btn.disabled = True
        self.stop_btn.disabled = False

        # Clear graph data
        self.hbo_data.clear()
        self.hbr_data.clear()
        self.current_hbo.clear()
        self.current_hbr.clear()
        self.ich_flags.clear()

        self._init_graph_plot()
        self._init_head_map()
        self._refresh_readouts()

        # Reset ICH status to monitoring
        self.ich_status_label.text = 'ICH STATUS: MONITORING'
        self._alert_color.rgba = (0.2, 0.6, 0.2, 1)  # Green

        if self.on_start_callback:
            self.on_start_callback()

    def _on_stop_pressed(self, instance):
        self.start_btn.disabled = False
        self.stop_btn.disabled = True

        # Set ICH status to idle
        self.ich_status_label.text = 'ICH STATUS: IDLE'
        self._alert_color.rgba = (0.3, 0.3, 0.3, 1)  # Gray

        if self.on_stop_callback:
            self.on_stop_callback()

    # ─────────────────────────────────────────────────────────────────────────────
    # Event Log
    # ─────────────────────────────────────────────────────────────────────────────

    def append_log(self, text: str):
        """Append text to event log and auto-scroll to bottom."""
        self.log_label.text += text
        # Auto-scroll to bottom
        self._log_scroll.scroll_y = 0

    def clear_log(self):
        """Clear the event log."""
        self.log_label.text = ''
