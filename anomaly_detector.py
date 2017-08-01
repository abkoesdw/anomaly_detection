import pickle
import numpy as np
import pandas as pd
import datetime as dt


class Predictor:

    def __init__(self, path):
        with open(path + 'model_weight.pkl', 'rb') as f:
            weight_ = pickle.load(f)

        with open(path + 'model_bias.pkl', 'rb') as f:
            bias_ = pickle.load(f)
        self.weight = weight_
        self.bias = bias_
        self.output = []
        self.status = []
        self.prev_status = []
        self.in_param = []
        self.out_param = []
        self.anomaly = dict()
        self.constant_inc = 0.01
        self.duration = []
        self.prev_duration = []
        self.duration_settings = []

    def forward_pass(self, input_):

        x = np.dot(input_, self.weight['dense_1'].astype(np.float)) + self.bias['dense_1'].reshape(1, -1)
        x = np.maximum(x, 0)

        x = np.dot(x, self.weight['dense_2'].astype(np.float)) + self.bias['dense_2'].reshape(1, -1)
        x = np.maximum(x, 0)

        x = np.dot(x, self.weight['dense_3'].astype(np.float)) + self.bias['dense_3'].reshape(1, -1)
        self.output = np.maximum(x, 0)

    def compute_status(self, actual_output):
        error_gain = 0.5
        error_input = error_gain * np.abs(self.output - actual_output) ** 1
        self.prev_status = self.status
        if error_input < self.in_param[0]:
            self.status = self.out_param[0]
        elif self.in_param[0] <= error_input <= self.in_param[1]:
            self.status = self.out_param[1]
        else:
            self.status = self.out_param[-1]

    def compute_duration(self):
        self.anomaly['low'] = - 1
        self.anomaly['medium'] = - 1
        self.anomaly['high'] = - 1
        if self.prev_status == self.status and self.status == self.out_param[-1]:
            self.duration = self.prev_duration + self.constant_inc
            self.prev_duration = self.duration
            self.prev_status = self.status
        else:
            self.duration = 0
            self.prev_status = self.status
            self.prev_duration = 0

        if self.duration <= self.duration_settings[0]:
            self.anomaly['low'] = 0.5
        elif self.duration_settings[0] <= self.duration <= self.duration_settings[1]:
            self.anomaly['medium'] = 0.5
        else:
            self.anomaly['high'] = 0.5


class Examples:

    def __init__(self, path, file):
        with open(path + 'anomaly_test.pkl', 'rb') as f:
            target_anomaly_temp = pickle.load(f)

        with open(path + 'anomaly_gt.pkl', 'rb') as f:
            gt_anomaly_temp = pickle.load(f)

        self.file = file
        self.target_anomaly = target_anomaly_temp[self.file][:, 10].reshape(-1, 1)
        gt_anomaly_ = gt_anomaly_temp[self.file]['ut_pv220']
        gt_anomaly = np.zeros_like(self.target_anomaly)
        gt_anomaly[gt_anomaly_[0]:gt_anomaly_[1]] = 1
        gt_anomaly[gt_anomaly_[2]:gt_anomaly_[3]] = 1

        self.gt_anomaly = gt_anomaly
        self.ts = 15
        self.column = dict()
        self.ultrasound = 'ut_pv220'
        self.unused_step = [2, 3, 10]
        self.command = ['cycle_phase',
                        'cycle_step',
                        'chamber_vacuum_valve_pv220']
        context_min = 15
        self.context_sec = context_min * 60
        self.left_context = int((1.0 / self.ts) * self.context_sec)
        self.right_context = 0
        self.window_size = int(2 * 150 / self.ts)
        self.time = []
        self.feature = []
        self.target = []
        self.generate_var()
        self.generate_feat()
        self.generate_target()

    def generate_var(self):
        time = ['time']
        df = pd.read_csv(self.file, low_memory=False)
        temp = df['cycle_phase']

        idx = []
        k = 0
        for phase in temp:
            if phase not in self.unused_step:
                idx = np.append(idx, k)
            k += 1

        idx = idx.astype(np.int)
        for var in time:
            x = df[var].values[0].split(' ')

            date_ = x[0]
            time_ = x[1]
            date_ = date_.split('-')

            time_ = time_.split(':')
            time_temp = [dt.datetime(int(date_[0]), int(date_[1]), int(date_[2]),
                                     int(time_[0]), int(time_[1]),
                                     int(time_[2])) + dt.timedelta(seconds=i) for i in range(len(df[var].values))]
            self.column[var] = time_temp[idx[0]:idx[-1] + 1][::self.ts]

        self.column[self.ultrasound] = df[self.ultrasound][idx][::self.ts]

        for var3 in self.command:
            self.column[var3] = df[var3][idx][::self.ts]

        self.time = np.array(self.column['time']).reshape(len(self.column['time']), 1)

    def generate_feat(self):
        keys = sorted(self.column.keys())
        num_data = np.shape(self.column[keys[0]])[0]
        feature_temp1 = np.empty((num_data, 0))
        for key in keys:
            if 'ut_' not in key and key != 'time':
                feature_temp1 = np.concatenate((feature_temp1, self.column[key].values.reshape(num_data, 1)), axis=1)

        _, num_feat = feature_temp1.shape
        feature_temp1 = np.concatenate((np.zeros((self.left_context, num_feat)), feature_temp1))
        feature_temp = np.empty((num_feat * (self.left_context + self.right_context + 1), 0))
        for j in range(self.left_context, num_data + self.left_context - self.right_context):
            current_frame = feature_temp1[j, :].reshape(num_feat, 1)
            left_frame = feature_temp1[j - self.left_context: j, :].reshape(num_feat * self.left_context, 1)
            total_frame = np.concatenate((left_frame, current_frame), axis=0)
            feature_temp = np.concatenate((feature_temp, total_frame), axis=1)

        self.feature = feature_temp.T

    def generate_target(self):
        num_data = np.shape(self.column[self.ultrasound])[0]
        target_temp = self.column[self.ultrasound].values.reshape(num_data, 1)

        target_temp_mean = np.concatenate((np.zeros((self.window_size - 1, 1)), np.copy(target_temp)))
        target = np.empty((1, 0))

        for k in range(0, len(target_temp)):
            target_temp2 = np.mean(target_temp_mean[k: k + self.window_size, 0])
            target = np.append(target, target_temp2)

        self.target = target.reshape(-1, 1)






