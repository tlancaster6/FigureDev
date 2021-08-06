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
                ax.plot(pm.countdown_times, pm.abs_volume_changes, 'r')
                if pm.abs_volume_changes[-1] == 0:
                    print('no volume change detected in behave trial {}'.format(pid))
            elif pm.trial_info.type.values[0] == 'C':
                ax.plot(pm.countdown_times, pm.abs_volume_changes, 'b')
                if pm.abs_volume_changes[-1] > 0:
                    print('volume change of {} detected in control trial {}'.format(pm.abs_volume_changes[-1], pid))
        ax.set(xlabel='time before euthanization, minutes', ylabel='cumulative absolute in-bower volume change (cm^3)')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'depth_change_comparison.pdf')
        plt.close(fig)

        path = self.output_dir / 'depth_change_comparison.csv'
        rows = []
        for pid, pm in self.project_managers.items():
            row = {'pid': pid}
            row.update({'behave_or_control': pm.trial_info.type.values[0]})
            row.update({'cumulative_abs_volume_change': pm.abs_volume_changes[-1]})
            for key, value in pm.trial_info.items():
                row.update({key: value.item()})
            row.update({'all_times_until_euthanization': [str.split(str(t), ' ')[0] for t in pm.countdown_times]})
            row.update({'all_cumulative_abs_volume_changes': pm.abs_volume_changes})
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(path)

    def plot_depth_change_comparison_v2(self):
        fig, ax = plt.subplots(1, 5, figsize=(11, 5), sharex='all', sharey='all')
        for pid, pm in self.project_managers.items():
            if pm.trial_info.type.values[0] == 'B':
                pool_name = pm.trial_info.pool.values[0]
                pool_number = int(pool_name[-1])
                ax[pool_number-1].plot(pm.countdown_times, pm.abs_volume_changes, 'r')
                ax[pool_number-1].set(xlabel='time before\neuthanization, minutes',
                                    ylabel='cumulative absolute in-bower volume change (cm^3)',
                                    title=pool_name)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'depth_change_comparison_v2.pdf')
        plt.close(fig)

    def download_all(self):
        for pid in self.trial_df.project_id.values:
            print('downloading ' + pid)
            cloud_project_dir = 'cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/' + pid + '/'
            local_project_dir = (self.data_dir/pid).as_posix()
            if not (self.data_dir/pid/'MasterAnalysisFiles').exists():
                (self.data_dir/pid/'MasterAnalysisFiles').mkdir(parents=True)
            if not (self.data_dir / pid / 'Troubleshooting').exists():
                (self.data_dir / pid / 'Troubleshooting').mkdir(parents=True)
            run(['rclone', 'copy',
                 cloud_project_dir + 'MasterAnalysisFiles/smoothedDepthData.npy',
                 local_project_dir+'/MasterAnalysisFiles/'])
            run(['rclone', 'copy',
                 cloud_project_dir + 'Troubleshooting/interpDepthData.npy',
                 local_project_dir+'/Troubleshooting/'])
            run(['rclone', 'copy',
                 cloud_project_dir + 'MasterAnalysisFiles/DepthCrop.txt',
                 local_project_dir+'/MasterAnalysisFiles/'])
            run(['rclone', 'copy',
                 cloud_project_dir + 'Logfile.txt',
                 local_project_dir])

    def re_calc_volume_changes(self, **kwargs):
        for pid, pm in self.project_managers.items():
            pm.abs_volume_changes = pm.calc_volume_changes(**kwargs)




class ProjectManager:

    def __init__(self, pid):
        self.pid = pid
        self.pixelLength = 0.1030168618
        self.home_dir = Path('D:') if Path('D:').exists() else Path.home()
        self.project_dir = self.home_dir / 'Temp' / 'SingleNuc' / pid
        self.project_summary_dir = self.project_dir / 'Summary'
        if not self.project_summary_dir.exists():
            self.project_summary_dir.mkdir(parents=True)
        self.lp = self.parse_log()
        self.trial_info = self.get_trial_info()
        self.tray_crop = self.load_tray_crop()
        self.smoothed_depth_data, self.actual_times, self.countdown_times = self.load_smoothed_depth_data()
        self.abs_volume_changes = self.calc_volume_changes()

    def parse_log(self):
        path = self.project_dir / 'Logfile.txt'
        lp = LP(path)
        return lp

    def get_trial_info(self):
        path = self.project_dir.parent / 'trials.csv'
        df = pd.read_csv(path, parse_dates=['dissection_time'], infer_datetime_format=True)
        return df.query('project_id == "{}"'.format(self.pid))

    def load_tray_crop(self):
        path = self.project_dir / 'MasterAnalysisFiles' / 'DepthCrop.txt'
        with open(path) as f:
            line = next(f)
            tray = line.rstrip().split(',')
            tray_crop = [int(x) for x in tray]
        return tray_crop

    def load_smoothed_depth_data(self, time_frame=np.timedelta64(90, 'm'), end_clip=np.timedelta64(10, 'm')):
        # path = self.project_dir / 'MasterAnalysisFiles' / 'SmoothedDepthData.npy'
        path = self.project_dir / 'Troubleshooting' / 'interpDepthData.npy'
        gc.collect()
        smoothed_depth_data = np.load(path)

        t1 = self.trial_info.dissection_time.values[0] - end_clip
        t0 = t1 - time_frame
        first_index = max([False if np.datetime64(x.time) <= t0 else True for x in self.lp.frames].index(True) - 1, 0)
        try:
            last_index = max([False if np.datetime64(x.time) <= t1 else True for x in self.lp.frames].index(True) - 1, 0)
        except ValueError:
            last_index = smoothed_depth_data.shape[0]
        while (t1 - np.datetime64(self.lp.frames[first_index].time)) > time_frame:
            first_index += 1
        trimmed_depth_data = smoothed_depth_data[first_index: last_index + 1]
        del smoothed_depth_data
        gc.collect()
        actual_times = [np.datetime64(x.time) for x in self.lp.frames[first_index: last_index+1]]
        countdown_times = [(x - self.trial_info.dissection_time.values[0]).astype('timedelta64[m]') for x in actual_times]
        return trimmed_depth_data, actual_times, countdown_times

    def calc_volume_changes(self, threshold=0.2, min_pixels=1000, trim_width=0, bower_mask=True):
        smd = self.smoothed_depth_data[:, self.tray_crop[0] + trim_width:self.tray_crop[2] - trim_width,
              self.tray_crop[1] + trim_width:self.tray_crop[3] - trim_width]
        changes = smd - smd[0]
        if bower_mask:
            castle_mask = np.where(changes[-1] >= threshold, True, False)
            castle_mask = morphology.remove_small_objects(castle_mask, min_pixels)
            pit_mask = np.where(changes[-1] <= -threshold, True, False)
            pit_mask = morphology.remove_small_objects(pit_mask, min_pixels)
            total_mask = castle_mask + pit_mask
            changes = changes * total_mask
        changes = np.abs(changes)
        changes = np.nansum(changes, axis=(1, 2))
        changes = changes * (self.pixelLength ** 2)
        return changes

    def plot_depth_detail(self, threshold=0.2, min_pixels=1000, trim_width=0, bower_mask=True):
        smd = self.smoothed_depth_data[:, self.tray_crop[0] + trim_width:self.tray_crop[2] - trim_width,
              self.tray_crop[1] + trim_width:self.tray_crop[3] - trim_width]
        changes = smd - smd[0]
        if bower_mask:
            castle_mask = np.where(changes[-1] >= threshold, True, False)
            castle_mask = morphology.remove_small_objects(castle_mask, min_pixels)
            pit_mask = np.where(changes[-1] <= -threshold, True, False)
            pit_mask = morphology.remove_small_objects(pit_mask, min_pixels)
            total_mask = castle_mask + pit_mask
            changes = changes * total_mask

        n_frames = self.smoothed_depth_data.shape[0]
        n_cols = 5
        n_rows = int(np.ceil(n_frames/n_cols))
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(11, 8.5))
        flax = ax.flatten()


        for i in range(n_frames):
            flax[i].imshow(changes[i], vmin=-1, vmax=1)
            t0 = pd.to_datetime(self.actual_times[0]).strftime('%H:%M')
            t1 = pd.to_datetime(self.actual_times[i]).strftime('%H:%M')
            flax[i].set(title='{}-{}'.format(t0, t1))
        for x in flax:
            x.set(xticks=[], yticks=[])
            sns.despine(ax=x, left=True, bottom=True)
        fig.tight_layout()
        fig.savefig(self.project_summary_dir/'DepthDetail.pdf')
        plt.close(fig=fig)



