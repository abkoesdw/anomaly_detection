import numpy as np
import glob
import matplotlib.pyplot as plt
import anomaly_detector as ad
import argparse

# input argument from command lines
parser = argparse.ArgumentParser()
parser.add_argument("file_id", help="insert file_id here (0 to 2)", type=int)
args = parser.parse_args()

# load the test files
files_test = sorted(glob.glob('./data_cycle_13/test/*.csv'))
file = files_test[args.file_id]

path = './'

# initialize example
example = ad.Examples(path, file)
feature = example.feature
target = example.target
target_anomaly = example.target_anomaly
gt_anomaly = example.gt_anomaly
time = example.time

# initialize variables
y_est = np.zeros_like(target)
error = np.zeros_like(target)
status = np.zeros_like(target)
duration = np.zeros_like(target)
anomaly = np.zeros((len(target), 3))

# initialize predictor
predictor = ad.Predictor(path)
predictor.in_param = [1, 2, 4]
predictor.out_param = [0, 4, 10]
predictor.duration_settings = [0.09, 0.25]

# loop for every new data
for t in range(len(time)):
    print(time[t])
    feature_temp = np.copy(feature[t, :].reshape(1, -1))
    predictor.forward_pass(feature_temp)
    y_est[t] = predictor.output

    predictor.compute_status(target_anomaly[t])
    predictor.compute_duration()

    # anomaly status
    anomaly[t, 0] = predictor.anomaly['low']
    anomaly[t, 1] = predictor.anomaly['medium']
    anomaly[t, 2] = predictor.anomaly['high']

# plot some results
plt.subplot(2, 1, 1)
plt.plot(time, target_anomaly, label='actual')
plt.plot(time, y_est, 'r', label='prediction')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(time, anomaly[:, 0], 'o', color='lime', markersize=14, label='normal')
plt.plot(time, anomaly[:, 1], 'o', color='orange', markersize=14, label='anomaly medium')
plt.plot(time, anomaly[:, 2], 'o', color='red', markersize=14, label='anomaly high')
plt.plot(time, gt_anomaly, ':', color='black', markersize=8, label='ground truth anomaly')
plt.ylim(ymin=-0.01)
plt.ylim(ymax=np.maximum(1, np.max(duration)))
plt.legend()
plt.grid()
plt.show()
