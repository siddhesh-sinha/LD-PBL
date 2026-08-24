import sys
import os
import json
import gc
import re
import warnings

import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QLabel, QFileDialog,
    QSpinBox, QFormLayout, QMessageBox, QGroupBox, QSplitter,
    QTextEdit, QCheckBox, QScrollArea, QTabWidget, QProgressBar,
    QTableWidget, QTableWidgetItem, QGridLayout, QRadioButton,
    QButtonGroup
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ============================================================
# CHECKBOX WIDGET
# ============================================================

class CheckBoxWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)


# ============================================================
# DATA PROCESSING WORKER
# ============================================================

class DataProcessorWorker(QObject):

    progress_updated = Signal(int)
    status_updated = Signal(str)
    processing_finished = Signal(object, str)
    error_occurred = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------

    def run(self):
        try:
            analysis_type = self.params["analysis_type"]

            is_dam = "DAM-based" in analysis_type

            file_path = (
                self.params["dam_file"]
                if is_dam
                else self.params["sorted_file"]
            )

            if not file_path:
                raise ValueError("Required data file path is missing.")

            if "Time-series" in analysis_type:
                self.process_time_series()
            else:
                self.process_batched_boxplot()

        except Exception as e:
            self.error_occurred.emit(
                f"Processing failed: {str(e)}"
            )

    # --------------------------------------------------------
    # FIND DEVELOPMENT ACTIVITY START
    # --------------------------------------------------------

    def find_confirmed_activity_starts(self, df):

        confirmed_starts = {}

        fly_columns = [
            col for col in df.columns
            if col.startswith("fly") and col.endswith("_x")
        ]

        dev_window = self.params.get("dev_window", 10)
        dev_movements = self.params.get("dev_movements", 1)

        for x_col in fly_columns:

            fly_num_str = x_col.replace("_x", "")
            y_col = f"{fly_num_str}_y"

            confirmed_starts[fly_num_str] = pd.NaT

            if y_col not in df.columns:
                continue

            valid_coord_mask = (
                df[x_col].notna()
                & (df[x_col] != 1)
                & df[y_col].notna()
                & (df[y_col] != 1)
            )

            valid_indices = df.index[valid_coord_mask]

            for idx in valid_indices:

                potential_start_time = df.loc[
                    idx, "timestamp_ist"
                ]

                window_end = (
                    potential_start_time
                    + pd.Timedelta(minutes=dev_window)
                )

                window_points = df[
                    (df["timestamp_ist"] > potential_start_time)
                    & (df["timestamp_ist"] <= window_end)
                ]

                if (
                    window_points[x_col].notna().sum()
                    >= dev_movements
                ):
                    confirmed_starts[fly_num_str] = (
                        potential_start_time
                    )
                    break

        return confirmed_starts

    # --------------------------------------------------------
    # DESICCATION TIME
    # --------------------------------------------------------

    def calculate_desiccation_time(
        self,
        fly_number,
        start_time,
        df_master
    ):

        x_col = f"fly{fly_number}_x"
        y_col = f"fly{fly_number}_y"

        if x_col not in df_master.columns:
            return None, None

        is_dam = "DAM-based" in self.params.get(
            "analysis_type", ""
        )

        time_col = (
            "start_time_ist"
            if is_dam
            else "timestamp_ist"
        )

        if time_col not in df_master.columns:
            return None, None

        cols = [x_col, time_col]

        if not is_dam and y_col in df_master.columns:
            cols.append(y_col)

        valid_data = df_master[cols].dropna()

        if valid_data.empty:
            return None, None

        valid_data = valid_data[
            valid_data[time_col] >= start_time
        ]

        if valid_data.empty:
            return None, None

        desiccation_timestamp = valid_data[
            time_col
        ].iloc[-1]

        divisor = (
            60.0
            if self.params.get("time_unit") == "Minutes"
            else 3600.0
        )

        duration = (
            desiccation_timestamp - start_time
        ).total_seconds() / divisor

        return duration, desiccation_timestamp

    # --------------------------------------------------------
    # NUMPY PROCESSING
    # --------------------------------------------------------

    def process_fly_data_numpy(
        self,
        fly_number,
        analysis_type,
        calc_start,
        df_master
    ):

        is_dam = "DAM-based" in analysis_type

        time_col = (
            "start_time_ist"
            if is_dam
            else "timestamp_ist"
        )

        if time_col not in df_master.columns:
            return None

        df_fly = df_master[
            df_master[time_col] >= calc_start
        ]

        if df_fly.empty:
            return None

        # ====================================================
        # DAM
        # ====================================================

        if is_dam:

            col = f"fly{fly_number}_count"

            if col not in df_fly.columns:
                return None

            counts = (
                df_fly[col]
                .fillna(0)
                .to_numpy()
            )

            if "Locomotor Activity" in analysis_type:
                return np.sum(counts)

            if "Resting Proportion" in analysis_type:

                zeros = np.sum(counts == 0)
                total = len(counts)

                return (
                    zeros / total
                    if total > 0
                    else 0
                )

            if "Average Speed" in analysis_type:

                total_activity = np.sum(counts)

                if total_activity == 0:
                    return 0

                cum_activity = np.cumsum(counts)

                target = total_activity * 0.99

                target_idx = np.argmax(
                    cum_activity >= target
                )

                time_diff = (
                    df_fly[time_col].iloc[target_idx]
                    - calc_start
                ).total_seconds()

                return (
                    target / time_diff
                    if time_diff > 0
                    else 0
                )

        # ====================================================
        # COORDINATES
        # ====================================================

        else:

            x_col = f"fly{fly_number}_x"
            y_col = f"fly{fly_number}_y"

            if x_col not in df_fly.columns:
                return None

            if y_col not in df_fly.columns:
                return None

            valid_df = df_fly[
                [x_col, y_col, time_col]
            ].dropna()

            if len(valid_df) < 2:
                return None

            pts = valid_df[
                [x_col, y_col]
            ].to_numpy()

            # ------------------------------------------------
            # RESTING PROPORTION
            # ------------------------------------------------

            if "Resting Proportion" in analysis_type:

                distances = np.linalg.norm(
                    np.diff(pts, axis=0),
                    axis=1
                )

                resting_frames = np.sum(
                    distances < 1.0
                )

                return (
                    resting_frames / len(pts)
                    if len(pts) > 0
                    else 0
                )

            # ------------------------------------------------
            # HOMOGRAPHY
            # ------------------------------------------------

            if "Length-based" in analysis_type:

                H = self.params.get("H")

                if H is None:
                    return None

                pts = cv2.perspectiveTransform(
                    pts.astype(np.float32)
                    .reshape(-1, 1, 2),
                    H
                ).reshape(-1, 2)

            distances = np.linalg.norm(
                np.diff(pts, axis=0),
                axis=1
            )

            total_dist = np.sum(distances)

            # ------------------------------------------------
            # LOCOMOTOR
            # ------------------------------------------------

            if "Locomotor Activity" in analysis_type:
                return total_dist

            # ------------------------------------------------
            # AVERAGE SPEED
            # ------------------------------------------------

            if "Average Speed" in analysis_type:

                if total_dist == 0:
                    return 0

                cum_dist = np.cumsum(
                    np.insert(distances, 0, 0)
                )

                target = total_dist * 0.99

                target_idx = np.argmax(
                    cum_dist >= target
                )

                time_diff = (
                    valid_df[time_col].iloc[target_idx]
                    - calc_start
                ).total_seconds()

                return (
                    target / time_diff
                    if time_diff > 0
                    else 0
                )

        return None

    # ========================================================
    # BATCHED BOXPLOT
    # ========================================================

    def process_batched_boxplot(self):

        analysis_type = self.params["analysis_type"]

        global_end_time = self.params["end_time"]

        groups = self.params["groups"]

        group_data_starts = (
            self.params["group_data_starts"]
        )

        group_analysis_starts = (
            self.params["group_analysis_starts"]
        )

        group_acclim = (
            self.params["group_acclimatization"]
        )

        available_cols = self.params["available_cols"]

        is_dam = "DAM-based" in analysis_type

        time_col = (
            "start_time_ist"
            if is_dam
            else "timestamp_ist"
        )

        file_path = (
            self.params["dam_file"]
            if is_dam
            else self.params["sorted_file"]
        )

        is_dev_analysis = (
            "Development Time" in analysis_type
        )

        is_des_analysis = (
            "Desiccation Time" in analysis_type
        )

        divisor = (
            60.0
            if self.params.get("time_unit") == "Minutes"
            else 3600.0
        )

        tasks = []

        for group_name, fly_list in groups.items():

            for fly in fly_list:
                tasks.append(
                    (group_name, fly)
                )

        total_tasks = len(tasks)

        if total_tasks == 0:
            self.error_occurred.emit(
                "No flies selected."
            )
            return

        batch_size = 40

        raw_data = []

        count = 0

        for i in range(
            0,
            total_tasks,
            batch_size
        ):

            batch_tasks = tasks[
                i:i + batch_size
            ]

            batch_flies = list(
                set(
                    fly
                    for _, fly in batch_tasks
                )
            )

            self.status_updated.emit(
                f"Loading Batch "
                f"{i // batch_size + 1}/"
                f"{(total_tasks + batch_size - 1) // batch_size}..."
            )

            # ------------------------------------------------
            # Columns
            # ------------------------------------------------

            cols_to_load = [time_col]

            for fly in batch_flies:

                if is_dam:

                    cols_to_load.append(
                        f"fly{fly}_count"
                    )

                else:

                    cols_to_load.extend([
                        f"fly{fly}_x",
                        f"fly{fly}_y"
                    ])

            valid_cols = [
                c for c in cols_to_load
                if c in available_cols
            ]

            dtypes = {
                col: "float32"
                for col in valid_cols
                if col != time_col
            }

            if time_col not in valid_cols:
                raise ValueError(
                    f"Time column '{time_col}' "
                    f"was not found in CSV."
                )

            # ------------------------------------------------
            # Read
            # ------------------------------------------------

            df_batch = pd.read_csv(
                file_path,
                usecols=valid_cols,
                dtype=dtypes,
                engine="c",
                encoding="utf-8-sig"
            )

            # ------------------------------------------------
            # Parse time
            # ------------------------------------------------

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    FutureWarning
                )

                if (
                    df_batch[time_col].dtype
                    == object
                ):

                    df_batch[time_col] = (
                        df_batch[time_col]
                        .astype(str)
                        .str.replace(
                            " IST",
                            "",
                            regex=False
                        )
                    )

                df_batch[time_col] = (
                    pd.to_datetime(
                        df_batch[time_col],
                        errors="coerce"
                    )
                )

                if (
                    df_batch[time_col].dt.tz
                    is not None
                ):

                    df_batch[time_col] = (
                        df_batch[time_col]
                        .dt.tz_localize(None)
                    )

            # ------------------------------------------------
            # Determine earliest DATA start in this batch
            # ------------------------------------------------

            batch_data_starts = []

            for group_name, _ in batch_tasks:

                start = group_data_starts.get(
                    group_name
                )

                if start is not None:
                    batch_data_starts.append(
                        start
                    )

            if not batch_data_starts:
                raise ValueError(
                    "No valid group Data Start times."
                )

            batch_min_start = min(
                batch_data_starts
            )

            # ------------------------------------------------
            # Time mask
            # ------------------------------------------------

            mask = (
                df_batch[time_col]
                >= batch_min_start
            )

            if global_end_time is not None:

                mask &= (
                    df_batch[time_col]
                    <= global_end_time
                )

            df_master = df_batch[
                mask
            ].copy()

            del df_batch
            gc.collect()

            # ------------------------------------------------
            # Development detection
            # ------------------------------------------------

            confirmed_starts = {}

            if is_dev_analysis:

                confirmed_starts = (
                    self.find_confirmed_activity_starts(
                        df_master
                    )
                )

            # ------------------------------------------------
            # Process each fly
            # ------------------------------------------------

            for group_name, fly in batch_tasks:

                count += 1

                progress = int(
                    (count / total_tasks) * 100
                )

                self.progress_updated.emit(
                    progress
                )

                self.status_updated.emit(
                    f"Calculated: "
                    f"{group_name}, "
                    f"Fly: {fly} "
                    f"({count}/{total_tasks})"
                )

                data_start = (
                    group_data_starts.get(
                        group_name
                    )
                )

                analysis_start = (
                    group_analysis_starts.get(
                        group_name
                    )
                )

                value = None

                calc_start_ts = pd.NaT
                calc_end_ts = pd.NaT

                # ====================================================
                # DESICCATION
                # ====================================================

                if is_des_analysis:

                    if analysis_start is None:
                        analysis_start = data_start

                    duration, desiccation_ts = (
                        self.calculate_desiccation_time(
                            fly,
                            analysis_start,
                            df_master
                        )
                    )

                    if duration is not None:

                        value = duration

                        calc_start_ts = (
                            analysis_start
                        )

                        calc_end_ts = (
                            desiccation_ts
                        )

                # ====================================================
                # DEVELOPMENT
                # ====================================================

                elif is_dev_analysis:

                    fly_id_str = f"fly{fly}"

                    confirmed_start = (
                        confirmed_starts.get(
                            fly_id_str,
                            pd.NaT
                        )
                    )

                    if (
                        not pd.isna(
                            confirmed_start
                        )
                        and analysis_start is not None
                    ):

                        duration = (
                            confirmed_start
                            - analysis_start
                        ).total_seconds() / divisor

                        if duration >= 0:

                            value = duration

                            calc_start_ts = (
                                analysis_start
                            )

                            calc_end_ts = (
                                confirmed_start
                            )

                # ====================================================
                # NORMAL ANALYSIS
                # ====================================================

                else:

                    acclim_min = (
                        group_acclim.get(
                            group_name,
                            0.0
                        )
                    )

                    calc_start = (
                        data_start
                        + pd.Timedelta(
                            minutes=acclim_min
                        )
                    )

                    value = (
                        self.process_fly_data_numpy(
                            fly,
                            analysis_type,
                            calc_start,
                            df_master
                        )
                    )

                    calc_start_ts = calc_start

                # ====================================================
                # DESICCATION EXCLUSION
                # ====================================================

                should_exclude = False

                if (
                    not is_des_analysis
                    and not is_dev_analysis
                ):

                    _, desiccation_ts_check = (
                        self.calculate_desiccation_time(
                            fly,
                            data_start,
                            df_master
                        )
                    )

                    if (
                        desiccation_ts_check is not None
                        and global_end_time is not None
                        and desiccation_ts_check
                        < global_end_time
                    ):

                        if self.params[
                            "exclude_desiccated"
                        ]:

                            should_exclude = True

                # ====================================================
                # SAVE RESULT
                # ====================================================

                if value is not None:

                    raw_data.append({

                        "group":
                            group_name,

                        "fly_id":
                            fly,

                        "value":
                            value,

                        "data_start_timestamp":
                            data_start,

                        "analysis_start_timestamp":
                            analysis_start,

                        "acclimatization_min":
                            group_acclim.get(
                                group_name,
                                0
                            ),

                        "excluded_due_to_desiccation":
                            should_exclude,

                        "start_timestamp_for_calc":
                            calc_start_ts,

                        "end_timestamp_for_calc":
                            calc_end_ts
                    })

            del df_master
            gc.collect()

        result_df = pd.DataFrame(
            raw_data
        )

        self.progress_updated.emit(100)

        if result_df.empty:

            raise ValueError(
                "No valid rows were produced for "
                "any fly/group.\n\n"
                "Check:\n"
                "1. Group Data Start times\n"
                "2. Group Analysis/Dev Start times\n"
                "3. CSV timestamps\n"
                "4. Fly numbers\n"
                "5. Start/End time range"
            )

        self.status_updated.emit(
            "All batches complete! "
            "Generating plot..."
        )

        self.processing_finished.emit(
            result_df,
            analysis_type
        )

    # ========================================================
    # TIME SERIES
    # ========================================================

    def process_time_series(self):

        fly1 = self.params["fly1"]
        fly2 = self.params["fly2"]

        use_homography = (
            self.params["use_homography"]
        )

        H = self.params.get("H")

        start_time = (
            self.params["start_time"]
        )

        end_time = (
            self.params["end_time"]
        )

        avg_window = (
            self.params["avg_window"]
        )

        analysis_type = (
            self.params["analysis_type"]
        )

        available_cols = (
            self.params["available_cols"]
        )

        is_dam = "DAM-based" in analysis_type

        time_col = (
            "start_time_ist"
            if is_dam
            else "timestamp_ist"
        )

        file_path = (
            self.params["dam_file"]
            if is_dam
            else self.params["sorted_file"]
        )

        self.status_updated.emit(
            "Extracting Time-Series data..."
        )

        cols_to_load = [time_col]

        for fly in [fly1, fly2]:

            if is_dam:

                cols_to_load.append(
                    f"fly{fly}_count"
                )

            else:

                cols_to_load.extend([
                    f"fly{fly}_x",
                    f"fly{fly}_y"
                ])

        valid_cols = [
            c for c in cols_to_load
            if c in available_cols
        ]

        dtypes = {
            col: "float32"
            for col in valid_cols
            if col != time_col
        }

        df_raw = pd.read_csv(
            file_path,
            usecols=valid_cols,
            dtype=dtypes,
            engine="c",
            encoding="utf-8-sig"
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                FutureWarning
            )

            if (
                df_raw[time_col].dtype
                == object
            ):

                df_raw[time_col] = (
                    df_raw[time_col]
                    .astype(str)
                    .str.replace(
                        " IST",
                        "",
                        regex=False
                    )
                )

            df_raw[time_col] = pd.to_datetime(
                df_raw[time_col],
                errors="coerce"
            )

            if (
                df_raw[time_col].dt.tz
                is not None
            ):

                df_raw[time_col] = (
                    df_raw[time_col]
                    .dt.tz_localize(None)
                )

        mask = (
            df_raw[time_col]
            >= start_time
        )

        if end_time is not None:

            mask &= (
                df_raw[time_col]
                <= end_time
            )

        df_master = df_raw[
            mask
        ].copy()

        del df_raw
        gc.collect()

        all_fly_dfs = []

        flies = [fly1, fly2]

        for i, fly_num in enumerate(flies):

            self.progress_updated.emit(
                int(
                    (i / len(flies))
                    * 100
                )
            )

            self.status_updated.emit(
                f"Processing Time-series "
                f"for Fly {fly_num}..."
            )

            x_col = f"fly{fly_num}_x"
            y_col = f"fly{fly_num}_y"

            if x_col not in df_master.columns:
                continue

            if y_col not in df_master.columns:
                continue

            df_fly_window = (
                df_master[
                    [
                        x_col,
                        y_col,
                        time_col
                    ]
                ]
                .dropna()
                .copy()
            )

            df_fly_window = (
                df_fly_window
                .drop_duplicates(
                    subset=time_col
                )
                .set_index(time_col)
            )

            if len(df_fly_window) < 2:
                continue

            pts = df_fly_window[
                [x_col, y_col]
            ].to_numpy()

            if (
                use_homography
                and H is not None
            ):

                pts = cv2.perspectiveTransform(
                    pts.astype(
                        np.float32
                    ).reshape(
                        -1, 1, 2
                    ),
                    H
                ).reshape(-1, 2)

            time_diff_s = (
                df_fly_window.index
                .to_series()
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )

            distance = np.linalg.norm(
                np.diff(
                    pts,
                    axis=0,
                    prepend=pts[0:1]
                ),
                axis=1
            )

            speed = pd.Series(
                np.divide(
                    distance,
                    time_diff_s,
                    out=np.zeros_like(
                        distance
                    ),
                    where=(
                        time_diff_s != 0
                    )
                ),
                index=df_fly_window.index
            )

            accel = pd.Series(
                np.divide(
                    speed.diff().fillna(0),
                    time_diff_s,
                    out=np.zeros_like(
                        speed.diff().fillna(0)
                    ),
                    where=(
                        time_diff_s != 0
                    )
                ),
                index=df_fly_window.index
            )

            avg_speed = (
                speed.rolling(
                    avg_window
                ).mean()
            )

            avg_accel = (
                accel.rolling(
                    avg_window
                ).mean()
            )

            unit = (
                "mm"
                if use_homography
                else "px"
            )

            all_fly_dfs.append(
                pd.DataFrame({

                    f"speed_{fly_num}_{unit}_s":
                        speed,

                    f"avg_speed_{fly_num}_{unit}_s":
                        avg_speed,

                    f"accel_{fly_num}_{unit}_s2":
                        accel,

                    f"avg_accel_{fly_num}_{unit}_s2":
                        avg_accel,

                    f"cum_dist_{fly_num}_{unit}":
                        np.cumsum(distance)
                })
            )

            _, desiccation_ts = (
                self.calculate_desiccation_time(
                    fly_num,
                    start_time,
                    df_master
                )
            )

            self.params[
                f"fly{fly_num}_desiccation"
            ] = desiccation_ts

        self.progress_updated.emit(100)

        if all_fly_dfs:

            result_df = pd.concat(
                all_fly_dfs,
                axis=1
            ).reset_index()

            result_df = (
                result_df.rename(
                    columns={
                        "index": time_col
                    }
                )
            )

        else:

            raise ValueError(
                "No valid rows were produced "
                "for either selected fly."
            )

        self.status_updated.emit(
            "Vectors computed. "
            "Generating plot..."
        )

        self.processing_finished.emit(
            result_df,
            "Time-series"
        )


# ============================================================
# MAIN GUI
# ============================================================

class FlyAnalysisGUI(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Fly Behavioral Analysis Tool"
        )

        self.sorted_file_path = None
        self.sorted_columns = []

        self.dam_file_path = None
        self.dam_columns = []

        self.groups = {}
        self.group_acclimatization = {}

        self.group_widgets = []

        self.exclusions = []

        self.H = None

        self.last_plot_data = None

        self.worker = None
        self.thread = None

        self.init_ui()

        self.add_group_widget(
            "Control",
            "1-10",
            "10"
        )

        self.add_group_widget(
            "Treated",
            "11-20",
            "10"
        )

    # ========================================================
    # UI
    # ========================================================

    def init_ui(self):

        main_widget = QWidget()

        self.setCentralWidget(
            main_widget
        )

        main_layout = QVBoxLayout(
            main_widget
        )

        tabs = QTabWidget()

        main_layout.addWidget(
            tabs
        )

        plot_tab = QWidget()

        self.setup_plot_tab(
            plot_tab
        )

        tabs.addTab(
            plot_tab,
            "Analysis & Plotting"
        )

        stats_tab = QWidget()

        self.setup_stats_tab(
            stats_tab
        )

        tabs.addTab(
            stats_tab,
            "Statistics"
        )

    # ========================================================
    # PLOT TAB
    # ========================================================

    def setup_plot_tab(self, plot_tab):

        plot_layout = QHBoxLayout(
            plot_tab
        )

        splitter = QSplitter(
            Qt.Horizontal
        )

        left_panel = QWidget()

        left_layout = QVBoxLayout(
            left_panel
        )

        left_layout.setAlignment(
            Qt.AlignTop
        )

        # ====================================================
        # FILE LOADING
        # ====================================================

        file_box = QGroupBox(
            "File Loading"
        )

        file_layout = QFormLayout()

        self.sorted_file_label = QLabel(
            "No coordinates CSV selected."
        )

        self.dam_file_label = QLabel(
            "No DAM CSV selected."
        )

        self.file_load_status = QLabel(
            "Ready"
        )

        self.file_load_status.setStyleSheet(
            "color: blue; font-style: italic;"
        )

        self.sorted_file_btn = QPushButton(
            "Map Coordinates CSV"
        )

        self.dam_file_btn = QPushButton(
            "Map DAM CSV"
        )

        self.sorted_file_btn.clicked.connect(
            lambda: self.map_csv("sorted")
        )

        self.dam_file_btn.clicked.connect(
            lambda: self.map_csv("dam")
        )

        file_layout.addRow(
            self.sorted_file_btn,
            self.sorted_file_label
        )

        file_layout.addRow(
            self.dam_file_btn,
            self.dam_file_label
        )

        file_layout.addRow(
            QLabel("Status:"),
            self.file_load_status
        )

        file_box.setLayout(
            file_layout
        )

        left_layout.addWidget(
            file_box
        )

        # ====================================================
        # CONFIGURATION SAVE / LOAD
        # ====================================================

        config_box = QGroupBox(
            "Group / Analysis Configuration"
        )

        config_layout = QHBoxLayout()

        self.save_config_btn = QPushButton(
            "Save Config (JSON)"
        )

        self.load_config_btn = QPushButton(
            "Load Config (JSON)"
        )

        self.save_config_btn.clicked.connect(
            self.save_config
        )

        self.load_config_btn.clicked.connect(
            self.load_config
        )

        config_layout.addWidget(
            self.save_config_btn
        )

        config_layout.addWidget(
            self.load_config_btn
        )

        config_box.setLayout(
            config_layout
        )

        left_layout.addWidget(
            config_box
        )

        # ====================================================
        # ANALYSIS
        # ====================================================

        analysis_box = QGroupBox(
            "Analysis Configuration"
        )

        analysis_layout = QFormLayout()

        self.analysis_combo = QComboBox()

        self.analysis_combo.addItems([
            "Locomotor Activity (Pixel-based)",
            "Locomotor Activity (Length-based)",
            "Locomotor Activity (DAM-based)",
            "Resting Proportion (DAM-based)",
            "Resting Proportion (Coordinates-based)",
            "Average Speed (Pixel-based)",
            "Average Speed (Length-based)",
            "Average Speed (DAM-based)",
            "Desiccation Time",
            "Development Time",
            "Time-series Plots (Comparison)"
        ])

        analysis_layout.addRow(
            "Analysis Type:",
            self.analysis_combo
        )

        analysis_box.setLayout(
            analysis_layout
        )

        left_layout.addWidget(
            analysis_box
        )

        # ====================================================
        # DEV / TIME OPTIONS
        # ====================================================

        dev_time_box = QGroupBox(
            "Time Units & Development Parameters"
        )

        dev_time_layout = QFormLayout()

        self.time_unit_combo = QComboBox()

        self.time_unit_combo.addItems([
            "Minutes",
            "Hours"
        ])

        self.dev_window_spin = QSpinBox()

        self.dev_window_spin.setRange(
            1,
            1000
        )

        self.dev_window_spin.setValue(
            10
        )

        self.dev_movements_spin = QSpinBox()

        self.dev_movements_spin.setRange(
            1,
            1000
        )

        self.dev_movements_spin.setValue(
            1
        )

        dev_time_layout.addRow(
            "Output Y-Axis Unit:",
            self.time_unit_combo
        )

        dev_time_layout.addRow(
            "Dev Window (min):",
            self.dev_window_spin
        )

        dev_time_layout.addRow(
            "Dev Minimum Movements:",
            self.dev_movements_spin
        )

        dev_time_box.setLayout(
            dev_time_layout
        )

        left_layout.addWidget(
            dev_time_box
        )

        # ====================================================
        # GLOBAL TIME RANGE
        # ====================================================

        time_box = QGroupBox(
            "Global Time Range"
        )

        time_layout = QFormLayout()

        self.start_time_edit = QLineEdit()

        self.start_time_edit.setPlaceholderText(
            "YYYY-MM-DD HH:MM:SS"
        )

        self.end_time_edit = QLineEdit()

        self.end_time_edit.setPlaceholderText(
            "YYYY-MM-DD HH:MM:SS"
        )

        self.interval_edit = QLineEdit()

        self.interval_edit.setPlaceholderText(
            "Optional e.g. 60 for 60 min"
        )

        self.start_time_label = QLabel(
            "Global Start Time:"
        )

        time_layout.addRow(
            self.start_time_label,
            self.start_time_edit
        )

        time_layout.addRow(
            QLabel("End Time:"),
            self.end_time_edit
        )

        time_layout.addRow(
            QLabel("Interval Limit:"),
            self.interval_edit
        )

        time_box.setLayout(
            time_layout
        )

        left_layout.addWidget(
            time_box
        )

        # ====================================================
        # GROUP MANAGEMENT
        # ====================================================

        self.group_box = QGroupBox(
            "Group Management"
        )

        self.group_box.setMinimumHeight(
            280
        )

        group_main_layout = QVBoxLayout()

        # ----------------------------------------------------
        # Per group start
        # ----------------------------------------------------

        self.per_group_start_check = QCheckBox(
            "Use different Start Times for each group"
        )

        self.per_group_start_check.stateChanged.connect(
            self.toggle_per_group_start
        )

        # ----------------------------------------------------
        # Scroll
        # ----------------------------------------------------

        scroll_area = QScrollArea()

        scroll_area.setWidgetResizable(
            True
        )

        self.group_list_widget = QWidget()

        self.group_list_layout = QVBoxLayout(
            self.group_list_widget
        )

        self.group_list_layout.setAlignment(
            Qt.AlignTop
        )

        scroll_area.setWidget(
            self.group_list_widget
        )

        # ----------------------------------------------------
        # Add group
        # ----------------------------------------------------

        add_group_btn = QPushButton(
            "Add Group"
        )

        add_group_btn.clicked.connect(
            lambda: self.add_group_widget()
        )

        group_main_layout.addWidget(
            self.per_group_start_check
        )

        group_main_layout.addWidget(
            scroll_area
        )

        group_main_layout.addWidget(
            add_group_btn
        )

        # ----------------------------------------------------
        # Exclusion
        # ----------------------------------------------------

        exclusion_layout = QFormLayout()

        self.exclusion_edit = QLineEdit()

        self.exclusion_edit.setPlaceholderText(
            "e.g. 5,8,12"
        )

        self.exclude_desiccated_check = QCheckBox(
            "Exclude pre-desiccated flies"
        )

        exclusion_layout.addRow(
            "Global Exclusions:",
            self.exclusion_edit
        )

        exclusion_layout.addRow(
            self.exclude_desiccated_check
        )

        group_main_layout.addLayout(
            exclusion_layout
        )

        self.group_box.setLayout(
            group_main_layout
        )

        left_layout.addWidget(
            self.group_box
        )

        # ====================================================
        # HOMOGRAPHY
        # ====================================================

        homography_box = QGroupBox(
            "Homography Settings"
        )

        homography_layout = QVBoxLayout()

        self.homography_paste_box = QTextEdit()

        self.homography_paste_box.setPlaceholderText(
            "Paste settings here..."
        )

        self.homography_paste_box.setMinimumHeight(
            50
        )

        update_homography_btn = QPushButton(
            "Update Homography from Pasted Text"
        )

        update_homography_btn.clicked.connect(
            self.parse_and_update_homography
        )

        homography_layout.addWidget(
            self.homography_paste_box
        )

        homography_layout.addWidget(
            update_homography_btn
        )

        homography_box.setLayout(
            homography_layout
        )

        left_layout.addWidget(
            homography_box
        )

        # ====================================================
        # TIME SERIES
        # ====================================================

        self.fly_select_box = QGroupBox(
            "Time-series Fly Selection"
        )

        fly_layout = QFormLayout()

        self.fly1_spin = QSpinBox()

        self.fly1_spin.setRange(
            1,
            240
        )

        self.fly1_spin.setValue(
            1
        )

        self.fly2_spin = QSpinBox()

        self.fly2_spin.setRange(
            1,
            240
        )

        self.fly2_spin.setValue(
            2
        )

        self.timeseries_plot_type_combo = QComboBox()

        self.timeseries_plot_type_combo.addItems([
            "Speed & Distance",
            "Acceleration"
        ])

        self.averaging_interval_edit = QLineEdit(
            "1.0"
        )

        self.use_homography_check = QCheckBox(
            "Use Homography"
        )

        self.use_homography_check.setChecked(
            True
        )

        fly_layout.addRow(
            "Fly 1:",
            self.fly1_spin
        )

        fly_layout.addRow(
            "Fly 2:",
            self.fly2_spin
        )

        fly_layout.addRow(
            "Plot Type:",
            self.timeseries_plot_type_combo
        )

        fly_layout.addRow(
            "Avg Interval (min):",
            self.averaging_interval_edit
        )

        fly_layout.addRow(
            self.use_homography_check
        )

        self.fly_select_box.setLayout(
            fly_layout
        )

        left_layout.addWidget(
            self.fly_select_box
        )

        # ====================================================
        # ACTIONS
        # ====================================================

        action_box = QGroupBox(
            "Actions"
        )

        action_layout = QVBoxLayout()

        self.plot_btn = QPushButton(
            "Generate Plot"
        )

        self.plot_btn.clicked.connect(
            self.generate_plot
        )

        self.save_plot_btn = QPushButton(
            "Save Plot"
        )

        self.save_plot_btn.clicked.connect(
            self.save_plot
        )

        self.save_data_btn = QPushButton(
            "Save Data"
        )

        self.save_data_btn.clicked.connect(
            self.save_data
        )

        self.plot_progress = QProgressBar()

        self.plot_progress.setRange(
            0,
            100
        )

        self.plot_progress.setValue(
            0
        )

        self.plot_status = QLabel(
            "Ready"
        )

        action_layout.addWidget(
            self.plot_btn
        )

        action_layout.addWidget(
            self.plot_progress
        )

        action_layout.addWidget(
            self.plot_status
        )

        action_layout.addWidget(
            self.save_plot_btn
        )

        action_layout.addWidget(
            self.save_data_btn
        )

        action_box.setLayout(
            action_layout
        )

        left_layout.addWidget(
            action_box
        )

        # ====================================================
        # RIGHT PLOT
        # ====================================================

        right_panel = QWidget()

        right_layout = QVBoxLayout(
            right_panel
        )

        self.figure = plt.figure(
            figsize=(12, 10)
        )

        self.canvas = FigureCanvas(
            self.figure
        )

        right_layout.addWidget(
            self.canvas
        )

        # ====================================================
        # SPLITTER
        # ====================================================

        left_scroll = QScrollArea()

        left_scroll.setWidgetResizable(
            True
        )

        left_scroll.setWidget(
            left_panel
        )

        left_scroll.setMinimumWidth(
            520
        )

        splitter.addWidget(
            left_scroll
        )

        splitter.addWidget(
            right_panel
        )

        splitter.setStretchFactor(
            0,
            0
        )

        splitter.setStretchFactor(
            1,
            1
        )

        plot_layout.addWidget(
            splitter
        )

        self.analysis_combo.currentIndexChanged.connect(
            self.update_ui_visibility
        )

        self.update_ui_visibility()

    # ========================================================
    # GROUP START TOGGLE
    # ========================================================

    def toggle_per_group_start(self):

        checked = (
            self.per_group_start_check.isChecked()
        )

        for w in self.group_widgets:

            w["data_start_label"].setVisible(
                checked
            )

            w["data_start"].setVisible(
                checked
            )

            w["analysis_start_label"].setVisible(
                checked
            )

            w["analysis_start"].setVisible(
                checked
            )

        self.start_time_label.setVisible(
            not checked
        )

        self.start_time_edit.setEnabled(
            not checked
        )

        self.update_ui_visibility()

    # ========================================================
    # STATS TAB
    # ========================================================

    def setup_stats_tab(self, stats_tab):

        grid_layout = QGridLayout(
            stats_tab
        )

        group_select_box = QGroupBox(
            "1. Select Groups for Comparison"
        )

        group_select_layout = QVBoxLayout()

        self.stats_summary_label = QLabel(
            "Generate a plot to populate groups "
            "for analysis."
        )

        self.stats_summary_label.setStyleSheet(
            "font-style: italic; color: grey;"
        )

        self.stats_group_table = QTableWidget(
            0,
            2
        )

        self.stats_group_table.setHorizontalHeaderLabels([
            "Group Name",
            "Compare?"
        ])

        self.stats_group_table.horizontalHeader().setStretchLastSection(
            True
        )

        group_select_layout.addWidget(
            self.stats_summary_label
        )

        group_select_layout.addWidget(
            self.stats_group_table
        )

        group_select_box.setLayout(
            group_select_layout
        )

        # ----------------------------------------------------

        test_select_box = QGroupBox(
            "2. Select and Run Test"
        )

        test_select_layout = QVBoxLayout()

        self.test_button_group = QButtonGroup(
            self
        )

        self.test_radios = {

            "Tukey HSD":
                QRadioButton(
                    "Tukey HSD (Multiple Groups)"
                ),

            "One-way ANOVA":
                QRadioButton(
                    "One-way ANOVA (Multiple Groups)"
                ),

            "Unpaired t-test":
                QRadioButton(
                    "Unpaired t-test (Exactly 2 Groups)"
                )
        }

        self.test_radios[
            "Tukey HSD"
        ].setChecked(True)

        for radio in self.test_radios.values():

            self.test_button_group.addButton(
                radio
            )

            test_select_layout.addWidget(
                radio
            )

        run_test_btn = QPushButton(
            "Run Test"
        )

        run_test_btn.clicked.connect(
            self.run_statistical_test
        )

        test_select_layout.addWidget(
            run_test_btn
        )

        test_select_box.setLayout(
            test_select_layout
        )

        # ----------------------------------------------------

        results_box = QGroupBox(
            "3. Statistical Test Results"
        )

        results_layout = QVBoxLayout()

        self.stats_results_table = QTableWidget()

        save_results_btn = QPushButton(
            "Save Test Results"
        )

        save_results_btn.clicked.connect(
            self.save_test_results
        )

        results_layout.addWidget(
            self.stats_results_table
        )

        results_layout.addWidget(
            save_results_btn
        )

        results_box.setLayout(
            results_layout
        )

        # ----------------------------------------------------

        stats_plot_box = QGroupBox(
            "4. Significance Plot"
        )

        stats_plot_layout = QVBoxLayout()

        self.stats_figure = plt.figure(
            figsize=(8, 6)
        )

        self.stats_canvas = FigureCanvas(
            self.stats_figure
        )

        stats_plot_layout.addWidget(
            self.stats_canvas
        )

        stats_plot_box.setLayout(
            stats_plot_layout
        )

        grid_layout.addWidget(
            group_select_box,
            0,
            0
        )

        grid_layout.addWidget(
            test_select_box,
            0,
            1
        )

        grid_layout.addWidget(
            stats_plot_box,
            1,
            0
        )

        grid_layout.addWidget(
            results_box,
            1,
            1
        )

        grid_layout.setRowStretch(
            0,
            1
        )

        grid_layout.setRowStretch(
            1,
            2
        )

    # ========================================================
    # STATS POPULATION
    # ========================================================

    def populate_stats_tab(self):

        self.stats_group_table.setRowCount(
            0
        )

        if (
            self.last_plot_data is None
            or self.last_plot_data.empty
        ):
            return

        analysis_type = (
            self.analysis_combo.currentText()
        )

        start_time_str = (
            self.start_time_edit.text()
        )

        end_time_str = (
            self.end_time_edit.text()
        )

        summary = (
            f"Analysis: {analysis_type} | "
            f"Global Start: {start_time_str}"
        )

        if end_time_str:

            summary += (
                f" | End: {end_time_str}"
            )

        self.stats_summary_label.setText(
            summary
        )

        self.stats_summary_label.setStyleSheet(
            ""
        )

        if (
            "excluded_due_to_desiccation"
            in self.last_plot_data.columns
        ):

            plotted_groups = (
                self.last_plot_data[
                    ~self.last_plot_data[
                        "excluded_due_to_desiccation"
                    ]
                ]["group"].unique()
            )

            self.stats_group_table.setRowCount(
                len(plotted_groups)
            )

            for i, group_name in enumerate(
                plotted_groups
            ):

                name_item = QTableWidgetItem(
                    group_name
                )

                name_item.setFlags(
                    name_item.flags()
                    & ~Qt.ItemIsEditable
                )

                cb_widget = CheckBoxWidget()

                cb_widget.checkbox.setChecked(
                    True
                )

                self.stats_group_table.setItem(
                    i,
                    0,
                    name_item
                )

                self.stats_group_table.setCellWidget(
                    i,
                    1,
                    cb_widget
                )

            self.stats_group_table.resizeColumnsToContents()

    # ========================================================
    # STATISTICAL TEST
    # ========================================================

    def run_statistical_test(self):

        if self.last_plot_data is None:
            return

        selected_groups = []

        for i in range(
            self.stats_group_table.rowCount()
        ):

            widget = (
                self.stats_group_table
                .cellWidget(i, 1)
            )

            if (
                widget is not None
                and widget.checkbox.isChecked()
            ):

                selected_groups.append(
                    self.stats_group_table
                    .item(i, 0)
                    .text()
                )

        if len(selected_groups) < 2:

            QMessageBox.warning(
                self,
                "Statistics",
                "Please select at least "
                "two groups."
            )

            return

        df_test = self.last_plot_data[
            (
                ~self.last_plot_data[
                    "excluded_due_to_desiccation"
                ]
            )
            &
            (
                self.last_plot_data[
                    "group"
                ].isin(selected_groups)
            )
        ].copy()

        test_name = (
            self.test_button_group
            .checkedButton()
            .text()
        )

        results_df = None

        try:

            # ------------------------------------------------
            # T TEST
            # ------------------------------------------------

            if "Unpaired t-test" in test_name:

                if len(selected_groups) != 2:

                    raise ValueError(
                        "Unpaired t-test requires "
                        "exactly 2 groups."
                    )

                group1_data = (
                    df_test[
                        df_test["group"]
                        == selected_groups[0]
                    ]["value"].dropna()
                )

                group2_data = (
                    df_test[
                        df_test["group"]
                        == selected_groups[1]
                    ]["value"].dropna()
                )

                stat, p_val = ttest_ind(
                    group1_data,
                    group2_data,
                    equal_var=False
                )

                cohens_d = (
                    self.calculate_cohens_d(
                        group1_data,
                        group2_data
                    )
                )

                results_df = pd.DataFrame([{

                    "Group 1":
                        selected_groups[0],

                    "Group 2":
                        selected_groups[1],

                    "p-value":
                        f"{p_val:.4f}",

                    "Significant (p<0.05)":
                        (
                            "Yes"
                            if p_val < 0.05
                            else "No"
                        ),

                    "Cohen's d":
                        f"{cohens_d:.3f}"
                }])

            # ------------------------------------------------
            # ANOVA
            # ------------------------------------------------

            elif "One-way ANOVA" in test_name:

                data_by_group = [
                    df_test[
                        df_test["group"]
                        == g
                    ]["value"].dropna()
                    for g in selected_groups
                ]

                f_val, p_val = f_oneway(
                    *data_by_group
                )

                results_df = pd.DataFrame([{

                    "Comparison":
                        "Overall",

                    "F-statistic":
                        f"{f_val:.3f}",

                    "p-value":
                        f"{p_val:.4f}",

                    "Significant (p<0.05)":
                        (
                            "Yes"
                            if p_val < 0.05
                            else "No"
                        )
                }])

            # ------------------------------------------------
            # TUKEY
            # ------------------------------------------------

            elif "Tukey HSD" in test_name:

                tukey_result = (
                    pairwise_tukeyhsd(
                        endog=df_test["value"],
                        groups=df_test["group"],
                        alpha=0.05
                    )
                )

                results_df = pd.DataFrame(
                    data=tukey_result
                    ._results_table
                    .data[1:],
                    columns=tukey_result
                    ._results_table
                    .data[0]
                )

                results_df = results_df.rename(
                    columns={
                        "p-adj": "p-value",
                        "reject":
                            "Significant (p<0.05)"
                    }
                )

                results_df[
                    "Significant (p<0.05)"
                ] = results_df[
                    "Significant (p<0.05)"
                ].replace({
                    True: "Yes",
                    False: "No"
                })

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )

            return

        if results_df is not None:

            self.stats_results_table.setRowCount(
                results_df.shape[0]
            )

            self.stats_results_table.setColumnCount(
                results_df.shape[1]
            )

            self.stats_results_table.setHorizontalHeaderLabels(
                results_df.columns
            )

            for r in range(
                results_df.shape[0]
            ):

                for c in range(
                    results_df.shape[1]
                ):

                    self.stats_results_table.setItem(
                        r,
                        c,
                        QTableWidgetItem(
                            str(
                                results_df.iloc[
                                    r,
                                    c
                                ]
                            )
                        )
                    )

            self.stats_results_table.resizeColumnsToContents()

            self.plot_stats_results(
                test_name,
                results_df
            )

    # ========================================================
    # STAT PLOT
    # ========================================================

    def plot_stats_results(
        self,
        test_name,
        data
    ):

        self.stats_figure.clear()

        ax = self.stats_figure.add_subplot(
            111
        )

        if (
            "Tukey HSD" in test_name
            and "group1" in data.columns
        ):

            groups = pd.unique(
                data[
                    ["group1", "group2"]
                ].values.ravel("K")
            )

            p_matrix = pd.DataFrame(
                np.ones(
                    (
                        len(groups),
                        len(groups)
                    )
                ),
                index=groups,
                columns=groups
            )

            for _, row in data.iterrows():

                p_val = float(
                    row["p-value"]
                )

                p_matrix.loc[
                    row["group1"],
                    row["group2"]
                ] = p_val

                p_matrix.loc[
                    row["group2"],
                    row["group1"]
                ] = p_val

            im = ax.imshow(
                p_matrix,
                cmap="Reds_r",
                vmin=0,
                vmax=0.1
            )

            self.stats_figure.colorbar(
                im,
                ax=ax,
                label="Adjusted p-value"
            )

            ax.set_xticks(
                np.arange(len(groups))
            )

            ax.set_yticks(
                np.arange(len(groups))
            )

            ax.set_xticklabels(
                groups
            )

            ax.set_yticklabels(
                groups
            )

            plt.setp(
                ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor"
            )

            for i in range(
                len(groups)
            ):

                for j in range(
                    len(groups)
                ):

                    if i == j:
                        continue

                    p_val = (
                        p_matrix.iloc[
                            i,
                            j
                        ]
                    )

                    text_color = (
                        "white"
                        if p_val < 0.05
                        else "black"
                    )

                    ax.text(
                        j,
                        i,
                        f"{p_val:.3f}",
                        ha="center",
                        va="center",
                        color=text_color
                    )

            ax.set_title(
                "Pairwise P-Value Heatmap"
            )

        else:

            ax.text(
                0.5,
                0.5,
                "No specific plot for "
                "this test.\nSee results table.",
                ha="center"
            )

        self.stats_figure.tight_layout()

        self.stats_canvas.draw()

    # ========================================================
    # SAVE TEST RESULTS
    # ========================================================

    def save_test_results(self):

        if (
            self.stats_results_table
            .rowCount()
            == 0
        ):
            return

        file_name, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Save Test Results",
                "",
                "CSV Files (*.csv)"
            )
        )

        if file_name:

            headers = [
                self.stats_results_table
                .horizontalHeaderItem(i)
                .text()
                for i in range(
                    self.stats_results_table
                    .columnCount()
                )
            ]

            data = [
                [
                    self.stats_results_table
                    .item(r, c)
                    .text()
                    for c in range(
                        self.stats_results_table
                        .columnCount()
                    )
                ]
                for r in range(
                    self.stats_results_table
                    .rowCount()
                )
            ]

            df = pd.DataFrame(
                data,
                columns=headers
            )

            df.to_csv(
                file_name,
                index=False
            )

    # ========================================================
    # COHEN'S D
    # ========================================================

    def calculate_cohens_d(
        self,
        g1,
        g2
    ):

        if len(g1) < 2 or len(g2) < 2:
            return np.nan

        m1 = np.mean(g1)
        s1 = np.std(g1, ddof=1)
        n1 = len(g1)

        m2 = np.mean(g2)
        s2 = np.std(g2, ddof=1)
        n2 = len(g2)

        pooled_std = np.sqrt(
            (
                (n1 - 1) * s1 ** 2
                +
                (n2 - 1) * s2 ** 2
            )
            /
            (
                n1 + n2 - 2
            )
        )

        return (
            (m1 - m2) / pooled_std
            if pooled_std > 0
            else 0
        )

    # ========================================================
    # ADD GROUP
    # ========================================================

    def add_group_widget(
        self,
        name="",
        range_str="",
        acclim_min="0",
        data_start="",
        analysis_start=""
    ):

        group_row_widget = QWidget()

        row_layout = QHBoxLayout(
            group_row_widget
        )

        row_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        name_edit = QLineEdit(
            name
        )

        name_edit.setPlaceholderText(
            "Group"
        )

        # ----------------------------------------------------
        # Range
        # ----------------------------------------------------

        range_edit = QLineEdit(
            range_str
        )

        range_edit.setPlaceholderText(
            "1-10"
        )

        # ----------------------------------------------------
        # Data Start
        # ----------------------------------------------------

        data_start_label = QLabel(
            "Data Start:"
        )

        data_start_edit = QLineEdit(
            data_start
        )

        data_start_edit.setPlaceholderText(
            "YYYY-MM-DD HH:MM:SS"
        )

        # ----------------------------------------------------
        # Analysis Start
        # ----------------------------------------------------

        analysis_start_label = QLabel(
            "Dev/Analysis Start:"
        )

        analysis_start_edit = QLineEdit(
            analysis_start
        )

        analysis_start_edit.setPlaceholderText(
            "YYYY-MM-DD HH:MM:SS"
        )

        # ----------------------------------------------------
        # Acclimatization
        # ----------------------------------------------------

        acclim_label = QLabel(
            "Acclim:"
        )

        acclim_edit = QLineEdit(
            str(acclim_min)
        )

        acclim_edit.setFixedWidth(
            55
        )

        # ----------------------------------------------------
        # Remove
        # ----------------------------------------------------

        remove_btn = QPushButton(
            "Remove"
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        row_layout.addWidget(
            QLabel("Name:")
        )

        row_layout.addWidget(
            name_edit
        )

        row_layout.addWidget(
            QLabel("Range:")
        )

        row_layout.addWidget(
            range_edit
        )

        row_layout.addWidget(
            data_start_label
        )

        row_layout.addWidget(
            data_start_edit
        )

        row_layout.addWidget(
            analysis_start_label
        )

        row_layout.addWidget(
            analysis_start_edit
        )

        row_layout.addWidget(
            acclim_label
        )

        row_layout.addWidget(
            acclim_edit
        )

        row_layout.addWidget(
            remove_btn
        )

        self.group_list_layout.addWidget(
            group_row_widget
        )

        widget_data = {

            "widget":
                group_row_widget,

            "name":
                name_edit,

            "range":
                range_edit,

            "acclim":
                acclim_edit,

            "acclim_label":
                acclim_label,

            "data_start_label":
                data_start_label,

            "data_start":
                data_start_edit,

            "analysis_start_label":
                analysis_start_label,

            "analysis_start":
                analysis_start_edit
        }

        self.group_widgets.append(
            widget_data
        )

        remove_btn.clicked.connect(
            lambda: self.remove_group_widget(
                widget_data
            )
        )

        # ----------------------------------------------------
        # Default values
        # ----------------------------------------------------

        if not data_start:

            data_start_edit.setText(
                self.start_time_edit.text()
            )

        if not analysis_start:

            analysis_start_edit.setText(
                self.start_time_edit.text()
            )

        self.update_ui_visibility()

    # ========================================================
    # REMOVE GROUP
    # ========================================================

    def remove_group_widget(
        self,
        widget_data
    ):

        if widget_data in self.group_widgets:

            self.group_widgets.remove(
                widget_data
            )

        widget_data[
            "widget"
        ].deleteLater()

    # ========================================================
    # UI VISIBILITY
    # ========================================================

    def update_ui_visibility(self):

        analysis_type = (
            self.analysis_combo.currentText()
        )

        is_timeseries = (
            "Time-series" in analysis_type
        )

        is_des_or_dev = (
            "Desiccation" in analysis_type
            or
            "Development Time" in analysis_type
        )

        is_checked = (
            self.per_group_start_check.isChecked()
        )

        self.group_box.setVisible(
            not is_timeseries
        )

        self.fly_select_box.setVisible(
            is_timeseries
        )

        self.start_time_label.setVisible(
            not is_checked
        )

        self.start_time_edit.setEnabled(
            not is_checked
        )

        self.end_time_edit.setVisible(
            not is_des_or_dev
        )

        self.interval_edit.setVisible(
            not is_des_or_dev
        )

        self.exclude_desiccated_check.setVisible(
            not is_des_or_dev
            and
            not is_timeseries
        )

        for group_widget in self.group_widgets:

            group_widget[
                "acclim"
            ].setVisible(
                not is_des_or_dev
            )

            group_widget[
                "acclim_label"
            ].setVisible(
                not is_des_or_dev
            )

            group_widget[
                "data_start_label"
            ].setVisible(
                is_checked
            )

            group_widget[
                "data_start"
            ].setVisible(
                is_checked
            )

            group_widget[
                "analysis_start_label"
            ].setVisible(
                is_checked
            )

            group_widget[
                "analysis_start"
            ].setVisible(
                is_checked
            )

    # ========================================================
    # MAP CSV
    # ========================================================

    def map_csv(
        self,
        csv_type
    ):

        file_name, _ = (
            QFileDialog.getOpenFileName(
                self,
                f"Open {csv_type.upper()} CSV",
                "",
                "CSV Files (*.csv)"
            )
        )

        if not file_name:
            return

        try:

            self.file_load_status.setText(
                f"Mapping {csv_type.upper()} file..."
            )

            QApplication.processEvents()

            cols = pd.read_csv(
                file_name,
                nrows=0,
                encoding="utf-8-sig"
            ).columns.tolist()

            if csv_type == "sorted":

                self.sorted_file_path = (
                    file_name
                )

                self.sorted_columns = cols

                self.sorted_file_label.setText(
                    os.path.basename(file_name)
                )

                time_col = (
                    "timestamp_ist"
                    if "timestamp_ist" in cols
                    else cols[1]
                )

            else:

                self.dam_file_path = (
                    file_name
                )

                self.dam_columns = cols

                self.dam_file_label.setText(
                    os.path.basename(file_name)
                )

                time_col = (
                    "start_time_ist"
                    if "start_time_ist" in cols
                    else cols[1]
                )

            # ------------------------------------------------
            # Boundary extraction
            # ------------------------------------------------

            self.file_load_status.setText(
                "Extracting Timeframes..."
            )

            QApplication.processEvents()

            first_val = None
            last_val = None

            # ------------------------------------------------
            # First row
            # ------------------------------------------------

            with open(
                file_name,
                "r",
                encoding="utf-8-sig"
            ) as f:

                header_line = f.readline()

                header = [
                    c.strip('"')
                    .strip("'")
                    .strip()
                    for c in header_line
                    .strip()
                    .split(",")
                ]

                if time_col in header:

                    col_idx = header.index(
                        time_col
                    )

                else:

                    col_idx = (
                        1
                        if len(header) > 1
                        else 0
                    )

                    time_col = header[
                        col_idx
                    ]

                first_data_line = (
                    f.readline()
                )

                if first_data_line:

                    parts = (
                        first_data_line
                        .strip()
                        .split(",")
                    )

                    if col_idx < len(parts):

                        first_val = (
                            parts[col_idx]
                            .strip('"')
                            .strip("'")
                            .strip()
                        )

            # ------------------------------------------------
            # Last row
            # ------------------------------------------------

            with open(
                file_name,
                "rb"
            ) as f:

                f.seek(
                    0,
                    os.SEEK_END
                )

                pos = f.tell()

                buffer = b""

                chunk_size = 4096

                found_last = False

                while (
                    pos > 0
                    and not found_last
                ):

                    if (
                        pos - chunk_size
                        < 0
                    ):
                        chunk_size = pos

                    pos -= chunk_size

                    f.seek(pos)

                    buffer = (
                        f.read(chunk_size)
                        + buffer
                    )

                    lines = (
                        buffer
                        .decode(
                            "utf-8-sig",
                            errors="ignore"
                        )
                        .splitlines()
                    )

                    for line in reversed(lines):

                        l_str = line.strip()

                        if not l_str:
                            continue

                        parts = (
                            l_str.split(",")
                        )

                        if col_idx < len(parts):

                            val = (
                                parts[col_idx]
                                .strip('"')
                                .strip("'")
                                .strip()
                            )

                            if (
                                val
                                and val != time_col
                            ):

                                last_val = val

                                found_last = True

                                break

            if not first_val or not last_val:

                raise ValueError(
                    "Could not locate boundary "
                    "timestamps."
                )

            # ------------------------------------------------
            # Parse
            # ------------------------------------------------

            first_val_clean = (
                str(first_val)
                .replace(" IST", "")
                .strip()
            )

            last_val_clean = (
                str(last_val)
                .replace(" IST", "")
                .strip()
            )

            dt_series = pd.to_datetime(
                [
                    first_val_clean,
                    last_val_clean
                ],
                errors="coerce"
            )

            if dt_series.isna().any():

                raise ValueError(
                    f"Extracted values "
                    f"['{first_val}', "
                    f"'{last_val}'] are not "
                    f"recognized as timestamps."
                )

            first_dt = (
                dt_series[0]
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            last_dt = (
                dt_series[1]
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            # ------------------------------------------------
            # Update global times
            # ------------------------------------------------

            self.start_time_edit.setText(
                first_dt
            )

            self.end_time_edit.setText(
                last_dt
            )

            # ------------------------------------------------
            # Update group times ONLY if empty
            # ------------------------------------------------

            for w in self.group_widgets:

                if not w[
                    "data_start"
                ].text().strip():

                    w[
                        "data_start"
                    ].setText(
                        first_dt
                    )

                if not w[
                    "analysis_start"
                ].text().strip():

                    w[
                        "analysis_start"
                    ].setText(
                        first_dt
                    )

            self.file_load_status.setText(
                f"{csv_type.upper()} File Mapped!"
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                f"Failed to map file: {e}"
            )

            self.file_load_status.setText(
                "Mapping failed."
            )

    # ========================================================
    # HOMOGRAPHY
    # ========================================================

    def parse_and_update_homography(self):

        text = (
            self.homography_paste_box
            .toPlainText()
        )

        try:

            coords_line = next(
                (
                    line.split(
                        ":",
                        1
                    )[1].strip()
                    for line in text.split("\n")
                    if line.lower()
                    .startswith("coords:")
                ),
                None
            )

            dists_line = next(
                (
                    line.split(
                        ":",
                        1
                    )[1].strip()
                    for line in text.split("\n")
                    if line.lower()
                    .startswith("dists (cm):")
                ),
                None
            )

            if not coords_line or not dists_line:

                raise ValueError(
                    "Could not find both "
                    "Coords and Dists lines."
                )

            coord_tuples = re.findall(
                r"\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)",
                coords_line
            )

            image_points = np.array(
                coord_tuples,
                dtype=np.float32
            )

            side_lengths_mm = [
                float(d.strip()) * 10
                for d in dists_line.split(",")
            ]

            real_width = (
                side_lengths_mm[0]
                + side_lengths_mm[2]
            ) / 2

            real_height = (
                side_lengths_mm[1]
                + side_lengths_mm[3]
            ) / 2

            world_points = np.array([
                [0, 0],
                [0, real_height],
                [real_width, real_height],
                [real_width, 0]
            ], dtype=np.float32)

            self.H, _ = cv2.findHomography(
                image_points,
                world_points
            )

            QMessageBox.information(
                self,
                "Success",
                "Homography matrix updated successfully."
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                f"Could not parse: {e}"
            )

            self.H = None

    # ========================================================
    # PARSE GROUPS
    # ========================================================

    def parse_groups_and_exclusions(self):

        try:

            excl_text = (
                self.exclusion_edit
                .text()
                .strip()
            )

            self.exclusions = [
                int(x.strip())
                for x in excl_text.split(",")
                if x.strip().isdigit()
            ]

            self.groups = {}

            self.group_acclimatization = {}

            if not self.group_widgets:

                cols = (
                    self.dam_columns
                    if self.dam_columns
                    else self.sorted_columns
                )

                if not cols:
                    return

                fly_numbers = sorted({
                    int(
                        c.split("_")[0]
                        .replace("fly", "")
                    )
                    for c in cols
                    if c.startswith("fly")
                })

                self.groups[
                    "All_Flies"
                ] = [
                    fly
                    for fly in fly_numbers
                    if fly not in self.exclusions
                ]

                self.group_acclimatization[
                    "All_Flies"
                ] = 0.0

                return

            # ------------------------------------------------
            # Parse each group
            # ------------------------------------------------

            for widget_data in self.group_widgets:

                name = (
                    widget_data["name"]
                    .text()
                    .strip()
                )

                range_str = (
                    widget_data["range"]
                    .text()
                    .strip()
                )

                acclim_str = (
                    widget_data["acclim"]
                    .text()
                    .strip()
                )

                if not name or not range_str:
                    continue

                try:

                    acclim_min = float(
                        acclim_str
                    )

                except ValueError:

                    acclim_min = 0.0

                fly_numbers = []

                for part in re.split(
                    r"[,\s]+",
                    range_str
                ):

                    part = part.strip()

                    if "-" in part:

                        start, end = map(
                            int,
                            part.split("-")
                        )

                        fly_numbers.extend(
                            range(
                                start,
                                end + 1
                            )
                        )

                    elif part.isdigit():

                        fly_numbers.append(
                            int(part)
                        )

                final_fly_list = [
                    fly
                    for fly in fly_numbers
                    if fly not in self.exclusions
                ]

                if final_fly_list:

                    self.groups[
                        name
                    ] = final_fly_list

                    self.group_acclimatization[
                        name
                    ] = acclim_min

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                f"Error parsing groups: {e}"
            )

            self.groups = {}

            self.group_acclimatization = {}

    # ========================================================
    # GET GLOBAL TIME RANGE
    # ========================================================

    def get_time_range(self):

        start_str = (
            self.start_time_edit
            .text()
            .strip()
        )

        end_str = (
            self.end_time_edit
            .text()
            .strip()
        )

        interval_min_str = (
            self.interval_edit
            .text()
            .strip()
        )

        try:

            start_time = (
                pd.to_datetime(
                    start_str
                )
                if start_str
                else None
            )

            end_time = None

            if end_str:

                end_time = pd.to_datetime(
                    end_str
                )

            elif (
                interval_min_str
                and start_time is not None
            ):

                end_time = (
                    start_time
                    + pd.Timedelta(
                        minutes=float(
                            interval_min_str
                        )
                    )
                )

            return (
                start_time,
                end_time
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                f"Invalid time selection: {e}"
            )

            return None, None

    # ========================================================
    # PARSE GROUP START TIMES
    # ========================================================

    def get_group_start_times(
        self,
        global_start
    ):

        group_data_starts = {}

        group_analysis_starts = {}

        per_group = (
            self.per_group_start_check
            .isChecked()
        )

        for w in self.group_widgets:

            name = (
                w["name"]
                .text()
                .strip()
            )

            if not name:
                continue

            # ------------------------------------------------
            # DATA START
            # ------------------------------------------------

            data_text = (
                w["data_start"]
                .text()
                .strip()
            )

            # ------------------------------------------------
            # ANALYSIS START
            # ------------------------------------------------

            analysis_text = (
                w["analysis_start"]
                .text()
                .strip()
            )

            try:

                if per_group and data_text:

                    data_start = pd.to_datetime(
                        data_text
                    )

                else:

                    data_start = global_start

            except Exception:

                raise ValueError(
                    f"Invalid Data Start "
                    f"for group '{name}'."
                )

            try:

                if per_group and analysis_text:

                    analysis_start = (
                        pd.to_datetime(
                            analysis_text
                        )
                    )

                else:

                    analysis_start = global_start

            except Exception:

                raise ValueError(
                    f"Invalid Analysis/Dev "
                    f"Start for group '{name}'."
                )

            if data_start is None:

                raise ValueError(
                    f"No Data Start specified "
                    f"for group '{name}'."
                )

            if analysis_start is None:

                analysis_start = data_start

            group_data_starts[
                name
            ] = data_start

            group_analysis_starts[
                name
            ] = analysis_start

        return (
            group_data_starts,
            group_analysis_starts
        )

    # ========================================================
    # GENERATE PLOT
    # ========================================================

    def generate_plot(self):

        analysis_type = (
            self.analysis_combo.currentText()
        )

        is_dam = (
            "DAM-based" in analysis_type
        )

        is_coord = any(
            x in analysis_type
            for x in [
                "Coordinates-based",
                "Pixel-based",
                "Length-based",
                "Time-series",
                "Desiccation",
                "Development Time"
            ]
        )

        if (
            is_dam
            and not self.dam_file_path
        ):

            QMessageBox.warning(
                self,
                "Error",
                "Please map the DAM CSV."
            )

            return

        if (
            is_coord
            and not self.sorted_file_path
        ):

            QMessageBox.warning(
                self,
                "Error",
                "Please map the Coordinates CSV."
            )

            return

        if (
            is_coord
            and self.H is None
            and any(
                x in analysis_type
                for x in [
                    "Length-based"
                ]
            )
        ):

            QMessageBox.warning(
                self,
                "Error",
                "Homography is not set."
            )

            return

        # ----------------------------------------------------
        # Groups
        # ----------------------------------------------------

        self.parse_groups_and_exclusions()

        if not self.groups:

            QMessageBox.warning(
                self,
                "Error",
                "No valid groups found."
            )

            return

        # ----------------------------------------------------
        # Global time
        # ----------------------------------------------------

        global_start, global_end = (
            self.get_time_range()
        )

        # ----------------------------------------------------
        # Group time
        # ----------------------------------------------------

        try:

            (
                group_data_starts,
                group_analysis_starts
            ) = self.get_group_start_times(
                global_start
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                str(e)
            )

            return

        # ----------------------------------------------------
        # Time-series averaging
        # ----------------------------------------------------

        try:

            avg_interval_min = float(
                self.averaging_interval_edit
                .text()
            )

            avg_window = (
                f"{int(avg_interval_min * 60)}s"
            )

        except ValueError:

            avg_window = "60s"
            avg_interval_min = 1.0

        available_cols = (
            self.dam_columns
            if is_dam
            else self.sorted_columns
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        params = {

            "analysis_type":
                analysis_type,

            "start_time":
                global_start,

            "end_time":
                global_end,

            "group_data_starts":
                group_data_starts,

            "group_analysis_starts":
                group_analysis_starts,

            "time_unit":
                self.time_unit_combo.currentText(),

            "dev_window":
                self.dev_window_spin.value(),

            "dev_movements":
                self.dev_movements_spin.value(),

            "sorted_file":
                self.sorted_file_path,

            "dam_file":
                self.dam_file_path,

            "available_cols":
                available_cols,

            "H":
                self.H,

            "groups":
                self.groups,

            "group_acclimatization":
                self.group_acclimatization,

            "exclude_desiccated":
                self.exclude_desiccated_check.isChecked(),

            "fly1":
                self.fly1_spin.value(),

            "fly2":
                self.fly2_spin.value(),

            "use_homography":
                self.use_homography_check.isChecked(),

            "avg_window":
                avg_window,

            "avg_interval_min":
                avg_interval_min
        }

        # ----------------------------------------------------
        # Start worker
        # ----------------------------------------------------

        self.plot_btn.setEnabled(
            False
        )

        self.plot_progress.setValue(
            0
        )

        self.figure.clear()

        self.thread = QThread()

        self.worker = DataProcessorWorker(
            params
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.progress_updated.connect(
            self.plot_progress.setValue
        )

        self.worker.status_updated.connect(
            self.plot_status.setText
        )

        self.worker.processing_finished.connect(
            self.on_processing_finished
        )

        self.worker.error_occurred.connect(
            self.on_processing_error
        )

        self.thread.start()

    # ========================================================
    # PROCESSING FINISHED
    # ========================================================

    def on_processing_finished(
        self,
        result_df,
        analysis_type
    ):

        self.last_plot_data = result_df

        if "Time-series" in analysis_type:

            self.render_timeseries_plot()

        else:

            self.render_boxplot(
                analysis_type
            )

        self.canvas.draw()

        self.plot_btn.setEnabled(
            True
        )

        self.cleanup_thread()

    # ========================================================
    # PROCESSING ERROR
    # ========================================================

    def on_processing_error(
        self,
        err_msg
    ):

        QMessageBox.critical(
            self,
            "Error",
            err_msg
        )

        self.plot_btn.setEnabled(
            True
        )

        self.plot_status.setText(
            "Failed."
        )

        self.cleanup_thread()

    # ========================================================
    # CLEAN THREAD
    # ========================================================

    def cleanup_thread(self):

        if self.thread:

            self.thread.quit()

            self.thread.wait()

            self.thread.deleteLater()

            if self.worker:

                self.worker.deleteLater()

            self.thread = None

            self.worker = None

    # ========================================================
    # Y LABEL
    # ========================================================

    def get_y_label(
        self,
        analysis_type
    ):

        if analysis_type in [
            "Desiccation Time",
            "Development Time"
        ]:

            return (
                f"{analysis_type.split(' ')[0]} "
                f"Time "
                f"({self.time_unit_combo.currentText()})"
            )

        labels = {

            "Locomotor Activity (Pixel-based)":
                "Distance (pixels)",

            "Locomotor Activity (Length-based)":
                "Distance (mm)",

            "Locomotor Activity (DAM-based)":
                "Total Activity (counts)",

            "Resting Proportion":
                "Resting Proportion (%)",

            "Average Speed (Pixel-based)":
                "Average Speed (pixels/s)",

            "Average Speed (Length-based)":
                "Average Speed (mm/s)",

            "Average Speed (DAM-based)":
                "Average Speed (counts/s)"
        }

        for key, value in labels.items():

            if key in analysis_type:

                return value

        return "Value"

    # ========================================================
    # RENDER BOXPLOT
    # ========================================================

    def render_boxplot(
        self,
        analysis_type
    ):

        ax = self.figure.add_subplot(
            111
        )

        if (
            self.last_plot_data is None
            or self.last_plot_data.empty
        ):

            ax.text(
                0.5,
                0.5,
                "No data to display.",
                ha="center",
                va="center"
            )

            return

        plot_df = self.last_plot_data[
            self.last_plot_data[
                "excluded_due_to_desiccation"
            ] == False
        ].copy()

        if plot_df.empty:

            ax.text(
                0.5,
                0.5,
                "No data to display "
                "after exclusions.",
                ha="center",
                va="center"
            )

            return

        if "Resting Proportion" in analysis_type:

            plot_df["plot_value"] = (
                plot_df["value"] * 100
            )

        else:

            plot_df["plot_value"] = (
                plot_df["value"]
            )

        group_names = list(
            self.groups.keys()
        )

        grouped_for_plot = (
            plot_df
            .groupby("group")[
                "plot_value"
            ]
            .apply(list)
            .reindex(
                group_names,
                fill_value=[]
            )
        )

        bp = ax.boxplot(
            grouped_for_plot,
            tick_labels=group_names,
            patch_artist=True,
            showfliers=False,
            positions=range(
                1,
                len(group_names) + 1
            )
        )

        colors = (
            matplotlib.colormaps[
                "viridis"
            ](
                np.linspace(
                    0,
                    1,
                    len(group_names)
                )
            )
        )

        for patch, color in zip(
            bp["boxes"],
            colors
        ):

            patch.set_facecolor(
                color
            )

        y_max = -np.inf

        for i, name in enumerate(
            group_names
        ):

            pos = i + 1

            data = (
                grouped_for_plot
                .get(
                    name,
                    []
                )
            )

            if not data:
                continue

            y = data

            x = np.random.normal(
                pos,
                0.04,
                size=len(data)
            )

            ax.scatter(
                x,
                y,
                alpha=0.6,
                color="black",
                s=15,
                zorder=3
            )

            try:

                box_top = (
                    bp["whiskers"][
                        i * 2 + 1
                    ]
                    .get_ydata()[1]
                )

            except (
                KeyError,
                IndexError
            ):

                box_top = max(data)

            if box_top > y_max:

                y_max = box_top

            ax.text(
                pos,
                box_top,
                f" n={len(y)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="blue"
            )

            # ------------------------------------------------
            # GROUP START TIME INFORMATION
            # ------------------------------------------------

            if (
                not self.per_group_start_check
                .isChecked()
            ):

                data_start_text = (
                    self.start_time_edit
                    .text()
                    .strip()
                )

                analysis_start_text = (
                    self.start_time_edit
                    .text()
                    .strip()
                )

            else:

                w = next(
                    (
                        item
                        for item
                        in self.group_widgets
                        if item["name"]
                        .text()
                        .strip()
                        == name
                    ),
                    None
                )

                if w:

                    data_start_text = (
                        w["data_start"]
                        .text()
                        .strip()
                    )

                    analysis_start_text = (
                        w["analysis_start"]
                        .text()
                        .strip()
                    )

                else:

                    data_start_text = ""
                    analysis_start_text = ""

            # Short display
            try:

                data_dt = pd.to_datetime(
                    data_start_text
                )

                data_display = (
                    data_dt.strftime(
                        "%m-%d %H:%M"
                    )
                )

            except Exception:

                data_display = (
                    data_start_text
                )

            try:

                analysis_dt = pd.to_datetime(
                    analysis_start_text
                )

                analysis_display = (
                    analysis_dt.strftime(
                        "%m-%d %H:%M"
                    )
                )

            except Exception:

                analysis_display = (
                    analysis_start_text
                )

            ax.text(
                pos,
                -0.13,
                (
                    f"Data: {data_display}\n"
                    f"Analysis: {analysis_display}"
                ),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="darkblue"
            )

        # ----------------------------------------------------
        # Y LIMIT
        # ----------------------------------------------------

        if np.isfinite(y_max):

            margin = (
                abs(y_max) * 0.15
                if y_max != 0
                else 1
            )

            ax.set_ylim(
                top=y_max + margin
            )

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        ax.set_title(
            analysis_type
        )

        ax.set_ylabel(
            self.get_y_label(
                analysis_type
            )
        )

        ax.set_xlabel(
            "Group"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6
        )

        # Extra bottom space for start labels
        self.figure.subplots_adjust(
            bottom=0.22
        )

        self.populate_stats_tab()

    # ========================================================
    # TIME SERIES PLOT
    # ========================================================

    def render_timeseries_plot(self):

        if (
            self.last_plot_data is None
            or self.last_plot_data.empty
        ):

            ax = self.figure.add_subplot(
                111
            )

            ax.text(
                0.5,
                0.5,
                "No data available.",
                ha="center",
                va="center"
            )

            return

        # Detect time column
        time_col = (
            "timestamp_ist"
            if "timestamp_ist"
            in self.last_plot_data.columns
            else "start_time_ist"
        )

        df = (
            self.last_plot_data
            .set_index(time_col)
        )

        fly1 = (
            self.fly1_spin.value()
        )

        fly2 = (
            self.fly2_spin.value()
        )

        plot_type = (
            self.timeseries_plot_type_combo
            .currentText()
        )

        use_homography = (
            self.use_homography_check
            .isChecked()
            and self.H is not None
        )

        unit = (
            "mm"
            if use_homography
            else "px"
        )

        try:

            avg_interval_min = float(
                self.averaging_interval_edit
                .text()
            )

        except ValueError:

            avg_interval_min = 1.0

        gs = self.figure.add_gridspec(
            2,
            1,
            hspace=0.4
        )

        axs = gs.subplots(
            sharex=True
        )

        self.figure.suptitle(
            f"Time-series Analysis ({unit})"
        )

        for i, fly_num in enumerate(
            [fly1, fly2]
        ):

            ax = axs[i]

            speed_col = (
                f"speed_{fly_num}_{unit}_s"
            )

            if speed_col not in df.columns:

                ax.text(
                    0.5,
                    0.5,
                    f"Fly {fly_num} "
                    "data not found",
                    ha="center"
                )

                continue

            speed = df[
                speed_col
            ]

            avg_speed = df[
                f"avg_speed_{fly_num}_{unit}_s"
            ]

            accel = df[
                f"accel_{fly_num}_{unit}_s2"
            ]

            avg_accel = df[
                f"avg_accel_{fly_num}_{unit}_s2"
            ]

            cum_dist = df[
                f"cum_dist_{fly_num}_{unit}"
            ]

            if plot_type == "Speed & Distance":

                ax.plot(
                    speed.index,
                    speed,
                    label="Inst. Speed",
                    color="lightcoral",
                    alpha=0.7
                )

                ax.plot(
                    avg_speed.index,
                    avg_speed,
                    label=(
                        f"Avg. Speed "
                        f"({avg_interval_min} min)"
                    ),
                    color="darkred"
                )

                ax.set_ylabel(
                    f"Speed ({unit}/s)"
                )

                ax.set_title(
                    f"Fly {fly_num} - "
                    "Speed & Distance"
                )

                ax_dist = ax.twinx()

                ax_dist.plot(
                    df.index,
                    cum_dist,
                    color="darkgreen",
                    linestyle="--",
                    label="Cum. Dist"
                )

                ax_dist.set_ylabel(
                    f"Cumulative Distance "
                    f"({unit})",
                    color="darkgreen"
                )

                ax_dist.tick_params(
                    axis="y",
                    labelcolor="darkgreen"
                )

            elif plot_type == "Acceleration":

                ax.plot(
                    accel.index,
                    accel,
                    label="Inst. Accel.",
                    color="skyblue",
                    alpha=0.7
                )

                ax.plot(
                    avg_accel.index,
                    avg_accel,
                    label=(
                        f"Avg. Accel. "
                        f"({avg_interval_min} min)"
                    ),
                    color="darkblue"
                )

                ax.set_ylabel(
                    f"Acceleration "
                    f"({unit}/s²)"
                )

                ax.set_title(
                    f"Fly {fly_num} - "
                    "Acceleration"
                )

                ax.axhline(
                    0,
                    color="black",
                    linewidth=0.5,
                    linestyle="--"
                )

            ax.legend(
                loc="upper left"
            )

            ax.grid(
                True,
                linestyle=":",
                alpha=0.6
            )

        axs[1].set_xlabel(
            "Time"
        )

        gs.tight_layout(
            self.figure
        )

    # ========================================================
    # SAVE PLOT
    # ========================================================

    def save_plot(self):

        if not self.figure.axes:

            QMessageBox.warning(
                self,
                "Error",
                "No plot to save."
            )

            return

        file_name, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                "",
                "PNG (*.png);;SVG (*.svg)"
            )
        )

        if file_name:

            try:

                self.figure.savefig(
                    file_name,
                    dpi=300,
                    bbox_inches="tight"
                )

                QMessageBox.information(
                    self,
                    "Success",
                    f"Plot saved to {file_name}"
                )

            except Exception as e:

                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Could not save: {e}"
                )

    # ========================================================
    # SAVE DATA
    # ========================================================

    def save_data(self):

        if (
            self.last_plot_data is None
            or self.last_plot_data.empty
        ):
            return

        file_name, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Save Data",
                "",
                "CSV (*.csv)"
            )
        )

        if file_name:

            try:

                df_to_save = (
                    self.last_plot_data
                    .copy()
                )

                if (
                    "plot_value"
                    in df_to_save.columns
                ):

                    df_to_save = (
                        df_to_save.drop(
                            columns=[
                                "plot_value"
                            ]
                        )
                    )

                for col in [
                    "data_start_timestamp",
                    "analysis_start_timestamp",
                    "start_timestamp_for_calc",
                    "end_timestamp_for_calc"
                ]:

                    if col in df_to_save.columns:

                        df_to_save[col] = (
                            df_to_save[col]
                            .apply(
                                lambda x:
                                x.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                if pd.notna(x)
                                else ""
                            )
                        )

                df_to_save.to_csv(
                    file_name,
                    index=False
                )

                QMessageBox.information(
                    self,
                    "Success",
                    f"Data saved to {file_name}"
                )

            except Exception as e:

                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Could not save: {e}"
                )

    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    def save_config(self):

        file_name, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Save Analysis Configuration",
                "",
                "JSON Files (*.json)"
            )
        )

        if not file_name:
            return

        try:

            groups_config = []

            for w in self.group_widgets:

                groups_config.append({

                    "name":
                        w["name"]
                        .text()
                        .strip(),

                    "range":
                        w["range"]
                        .text()
                        .strip(),

                    "acclimatization":
                        w["acclim"]
                        .text()
                        .strip(),

                    "data_start":
                        w["data_start"]
                        .text()
                        .strip(),

                    "analysis_start":
                        w["analysis_start"]
                        .text()
                        .strip()
                })

            config = {

                "version":
                    2,

                "analysis_type":
                    self.analysis_combo
                    .currentText(),

                "global_start":
                    self.start_time_edit
                    .text()
                    .strip(),

                "global_end":
                    self.end_time_edit
                    .text()
                    .strip(),

                "interval":
                    self.interval_edit
                    .text()
                    .strip(),

                "per_group_start":
                    self.per_group_start_check
                    .isChecked(),

                "exclusions":
                    self.exclusion_edit
                    .text()
                    .strip(),

                "exclude_desiccated":
                    self.exclude_desiccated_check
                    .isChecked(),

                "time_unit":
                    self.time_unit_combo
                    .currentText(),

                "dev_window":
                    self.dev_window_spin
                    .value(),

                "dev_movements":
                    self.dev_movements_spin
                    .value(),

                "groups":
                    groups_config,

                "fly1":
                    self.fly1_spin.value(),

                "fly2":
                    self.fly2_spin.value(),

                "timeseries_plot_type":
                    self.timeseries_plot_type_combo
                    .currentText(),

                "averaging_interval":
                    self.averaging_interval_edit
                    .text(),

                "use_homography":
                    self.use_homography_check
                    .isChecked(),

                "homography_text":
                    self.homography_paste_box
                    .toPlainText()
            }

            with open(
                file_name,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    config,
                    f,
                    indent=4
                )

            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Configuration saved to:\n"
                f"{file_name}"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not save configuration:\n{e}"
            )

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    def load_config(self):

        file_name, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Load Analysis Configuration",
                "",
                "JSON Files (*.json)"
            )
        )

        if not file_name:
            return

        try:

            with open(
                file_name,
                "r",
                encoding="utf-8"
            ) as f:

                config = json.load(f)

            # ------------------------------------------------
            # Basic settings
            # ------------------------------------------------

            if "analysis_type" in config:

                index = (
                    self.analysis_combo
                    .findText(
                        config[
                            "analysis_type"
                        ]
                    )
                )

                if index >= 0:

                    self.analysis_combo.setCurrentIndex(
                        index
                    )

            self.start_time_edit.setText(
                config.get(
                    "global_start",
                    ""
                )
            )

            self.end_time_edit.setText(
                config.get(
                    "global_end",
                    ""
                )
            )

            self.interval_edit.setText(
                config.get(
                    "interval",
                    ""
                )
            )

            self.exclusion_edit.setText(
                config.get(
                    "exclusions",
                    ""
                )
            )

            self.per_group_start_check.setChecked(
                config.get(
                    "per_group_start",
                    True
                )
            )

            self.exclude_desiccated_check.setChecked(
                config.get(
                    "exclude_desiccated",
                    False
                )
            )

            # ------------------------------------------------
            # Time settings
            # ------------------------------------------------

            time_unit = config.get(
                "time_unit",
                "Minutes"
            )

            index = (
                self.time_unit_combo
                .findText(
                    time_unit
                )
            )

            if index >= 0:

                self.time_unit_combo.setCurrentIndex(
                    index
                )

            self.dev_window_spin.setValue(
                int(
                    config.get(
                        "dev_window",
                        10
                    )
                )
            )

            self.dev_movements_spin.setValue(
                int(
                    config.get(
                        "dev_movements",
                        1
                    )
                )
            )

            # ------------------------------------------------
            # Time-series
            # ------------------------------------------------

            self.fly1_spin.setValue(
                int(
                    config.get(
                        "fly1",
                        1
                    )
                )
            )

            self.fly2_spin.setValue(
                int(
                    config.get(
                        "fly2",
                        2
                    )
                )
            )

            ts_type = config.get(
                "timeseries_plot_type",
                "Speed & Distance"
            )

            index = (
                self.timeseries_plot_type_combo
                .findText(
                    ts_type
                )
            )

            if index >= 0:

                self.timeseries_plot_type_combo.setCurrentIndex(
                    index
                )

            self.averaging_interval_edit.setText(
                str(
                    config.get(
                        "averaging_interval",
                        "1.0"
                    )
                )
            )

            self.use_homography_check.setChecked(
                config.get(
                    "use_homography",
                    True
                )
            )

            self.homography_paste_box.setPlainText(
                config.get(
                    "homography_text",
                    ""
                )
            )

            # ------------------------------------------------
            # Remove current groups
            # ------------------------------------------------

            for w in self.group_widgets:

                w["widget"].deleteLater()

            self.group_widgets.clear()

            # ------------------------------------------------
            # Recreate groups
            # ------------------------------------------------

            for group in config.get(
                "groups",
                []
            ):

                self.add_group_widget(

                    name=group.get(
                        "name",
                        ""
                    ),

                    range_str=group.get(
                        "range",
                        ""
                    ),

                    acclim_min=group.get(
                        "acclimatization",
                        "0"
                    ),

                    data_start=group.get(
                        "data_start",
                        config.get(
                            "global_start",
                            ""
                        )
                    ),

                    analysis_start=group.get(
                        "analysis_start",
                        config.get(
                            "global_start",
                            ""
                        )
                    )
                )

            # ------------------------------------------------
            # Refresh visibility
            # ------------------------------------------------

            self.update_ui_visibility()

            QMessageBox.information(
                self,
                "Configuration Loaded",
                f"Configuration loaded from:\n"
                f"{file_name}"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load configuration:\n{e}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Windows HiDPI
    os.environ.setdefault(
        "QT_ENABLE_HIGHDPI_SCALING",
        "1"
    )

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(
        sys.argv
    )

    window = FlyAnalysisGUI()

    window.showMaximized()

    sys.exit(
        app.exec()
    )