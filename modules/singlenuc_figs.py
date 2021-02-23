from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils.LogParser import LogParser as LP
import gc
import pickle
from subprocess import run
from skimage import morphology


class DataManager:

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.home_dir = Path('D:') if Path('D:').exists() else Path.home()
        self.data_dir = self.home_dir / 'Temp' / 'SingleNuc'
        self.output_dir = self.home_dir / 'Temp' / 'SingleNuc' / 'Figures'
        self.cache_file = self.home_dir / 'Temp' / 'SingleNuc' / 'cache.pkl'
        self.trial_df = self.load_trials_df()
        self.project_managers = None

    def load_trials_df(self):
        path = self.data_dir / 'trials.csv'
        df = pd.read_csv(path, parse_dates=['dissection_time'], infer_datetime_format=True)
        return df

    def initiate_project_managers(self):
        if self.use_cache and self.cache_file.exists():
            self.project_managers = pickle.load(open(self.cache_file, 'rb'))
            return
        project_managers = {}
        for pid in self.trial_df.project_id:
            print('loading {}'.format(pid))
            project_managers.update({pid: ProjectManager(pid)})
        pickle.dump(project_managers, open(self.cache_file, 'wb'))
        self.project_managers = project_managers
        return

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

    def download_all(self):
        for pid in self.trial_df.project_id.values:
            print('downloading ' + pid)
            cloud_project_dir = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/' + pid + '/'
            local_project_dir = (self.data_dir/pid).as_posix()
            if not (self.data_dir/pid/'MasterAnalysisFiles').exists():
                (self.data_dir/pid/'MasterAnalysisFiles').mkdir(parents=True)
            run(['rclone', 'copy',
                 cloud_project_dir + 'MasterAnalysisFiles/smoothedDepthData.npy',
                 local_project_dir+'/MasterAnalysisFiles/'])
            run(['rclone', 'copy',
                 cloud_project_dir + 'MasterAnalysisFiles/DepthCrop.txt',
                 local_project_dir+'/MasterAnalysisFiles/'])
            run(['rclone', 'copy',
                 cloud_project_dir + 'Logfile.txt',
                 local_project_dir])


class ProjectManager:

    def __init__(self, pid):
        self.pid = pid
        self.pixelLength = 0.1030168618
        self.home_dir = Path('D:') if Path('D:').exists() else Path.home()
        self.project_dir = self.home_dir / 'Temp' / 'SingleNuc' / pid
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
        smoothed_depth_data[:, :tray_crop[0] + 10, :] = np.nan
        smoothed_depth_data[:, tray_crop[2] - 10:, :] = np.nan
        smoothed_depth_data[:, :, :tray_crop[1] + 10] = np.nan
        smoothed_depth_data[:, :, tray_crop[3] - 10:] = np.nan

        t1 = self.trial_info.dissection_time.values[0] - np.timedelta64(10, 'm')
        t0 = t1 - np.timedelta64(2, 'h')
        first_index = max([False if np.datetime64(x.time) <= t0 else True for x in self.lp.frames].index(True) - 1, 0)
        last_index = max([False if np.datetime64(x.time) <= t1 else True for x in self.lp.frames].index(True) - 1, 0)
        trimmed_depth_data = smoothed_depth_data[first_index: last_index + 1]
        del smoothed_depth_data
        gc.collect()
        return trimmed_depth_data

    def calc_volume_changes(self, threshold=0.1, min_pixels=1000):
        changes = np.abs(self.smoothed_depth_data - self.smoothed_depth_data[0])
        mask = np.where(changes[-1] >= threshold, True, False)
        mask = morphology.remove_small_objects(mask, min_pixels)
        changes = changes*mask
        changes = np.nansum(changes, axis=(1, 2))
        changes = changes * (self.pixelLength ** 2)
        return changes


