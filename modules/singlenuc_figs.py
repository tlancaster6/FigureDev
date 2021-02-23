from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils.LogParser import LogParser as LP
import gc
import pickle


class DataManager:

    def __init__(self):
        self.home_dir = Path('D:') if Path('D:').exists() else Path.home()

        self.data_dir = self.home_dir / 'Temp' / 'SingleNuc'
        self.output_dir = self.home_dir / 'Temp' / 'SingleNuc' / 'Figures'
        self.cache_file = self.home_dir / 'Temp' / 'SingleNuc' / 'cache.pkl'
        self.trial_df = self.load_trials_df()
        self.project_managers = self.initiate_project_managers()

    def load_trials_df(self):
        path = self.data_dir / 'trials.csv'
        df = pd.read_csv(path, parse_dates=['dissection_time'], infer_datetime_format=True)
        return df

    def initiate_project_managers(self):
        if self.cache_file.exists():
            return pickle.load(open(self.cache_file, 'rb'))
        project_managers = {}
        for pid in self.trial_df.project_id:
            print('loading {}'.format(pid))
            project_managers.update({pid: ProjectManager(pid)})
        pickle.dump(project_managers, open(self.cache_file, 'wb'))
        return project_managers

    def plot_depth_change_comparison(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for pid, pm in self.project_managers.items():
            if pm.trial_info.type.values[0] == 'B':
                ax.plot(pm.times, pm.abs_volume_changes, 'r')
            elif pm.trial_info.type.values[0] == 'C':
                ax.plot(pm.times, pm.abs_volume_changes, 'b')
        ax.set(xlabel='time before euthanization, minutes', ylabel='cumulative absolute volume change (cm^3)')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'depth_change_comparison.pdf')
        plt.close(fig)



class ProjectManager:

    def __init__(self, pid):
        self.pid = pid
        self.pixelLength = 0.1030168618
        self.project_dir = Path('D:', 'Temp', 'SingleNuc', pid)
        self.lp = self.parse_log()
        self.trial_info = self.get_trial_info()
        self.smoothed_depth_data = self.load_smoothed_depth_data()
        self.abs_volume_changes = self.calc_volume_changes()
        self.times = -5 * np.arange(0, self.abs_volume_changes.size)[::-1]

    def parse_log(self):
        path = self.project_dir / 'Logfile.txt'
        lp = LP(path)
        return lp

    def get_trial_info(self):
        path = self.project_dir.parent / 'trials.csv'
        df = pd.read_csv(path, parse_dates=['dissection_time'], infer_datetime_format=True)
        return df.query('project_id == "{}"'.format(self.pid))

    def load_smoothed_depth_data(self):
        path = self.project_dir / 'MasterAnalysisFiles' / 'SmoothedDepthData.npy'
        gc.collect()
        smoothed_depth_data = np.load(path)

        path = self.project_dir / 'MasterAnalysisFiles' / 'DepthCrop.txt'
        with open(path) as f:
            line = next(f)
            tray = line.rstrip().split(',')
            tray_crop = [int(x) for x in tray]
        smoothed_depth_data[:, :tray_crop[0], :] = np.nan
        smoothed_depth_data[:, tray_crop[2]:, :] = np.nan
        smoothed_depth_data[:, :, :tray_crop[1]] = np.nan
        smoothed_depth_data[:, :, tray_crop[3]:] = np.nan

        t1 = self.trial_info.dissection_time.values[0] - np.timedelta64(10, 'm')
        t0 = t1 - np.timedelta64(2, 'h')
        first_index = max([False if np.datetime64(x.time) <= t0 else True for x in self.lp.frames].index(True) - 1, 0)
        last_index = max([False if np.datetime64(x.time) <= t1 else True for x in self.lp.frames].index(True) - 1, 0)
        trimmed_depth_data = smoothed_depth_data[first_index: last_index + 1]
        del smoothed_depth_data
        gc.collect()
        return trimmed_depth_data

    def calc_volume_changes(self):
        abs_depth_change_per_pixel = np.abs(np.diff(self.smoothed_depth_data, axis=0))
        abs_depth_change_per_frame = np.nansum(abs_depth_change_per_pixel, axis=(1, 2))
        abs_volume_change_per_frame = abs_depth_change_per_frame * self.pixelLength ** 2
        abs_volume_change_per_frame = np.insert(abs_volume_change_per_frame, 0, 0)
        return abs_volume_change_per_frame


