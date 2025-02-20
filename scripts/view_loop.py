import os
import sys
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图模块

# 获取命令行参数作为文件夹路径
if len(sys.argv) < 2:
    print("python3 view.py path_to_result")
    sys.exit(1)

folder_paths = sys.argv[1]

# 创建一个3D图形对象
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['c', 'm', 'y', 'k']
colors_random = [plt.cm.tab20(i) for i in range(20)]
colors = colors + colors_random
i = 0
graph_path = ''
frame_map = dict()
session_map = dict()
for folder_path in os.listdir(folder_paths):
    csv_file = os.path.join(folder_paths, folder_path + '/lidar_mapping_result/tum_vehicle_enu_pose.csv')
    if not os.path.exists(csv_file):
        continue
    x = []
    y = []
    z = []
    with open(csv_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            timestamp = float("{:.2f}".format(float(row[0])))
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))
            frame_map[str(timestamp)] = [float(row[1]), float(row[2]), float(row[3])]
            session_map[str(timestamp)] = i
    ax.plot(x, y, z, label=folder_path, color=colors[i % len(colors)],linewidth=2)
    i = i+1
    if os.path.exists(folder_paths + '/' + folder_path + '/lidar_mapping_result/optimized_graph.txt'):
      graph_path = os.path.join(folder_paths, folder_path + '/lidar_mapping_result/optimized_graph.txt')
f = open(graph_path)
lines = f.readlines()
time2index = dict()
for i in range(len(lines)):
    if i == 0:
        continue      
    line_list = lines[i].split('\t')
    if line_list[0] == 'VERTEX_SE3' :
        value = line_list[1].split(' ')
        index = value[0]
        value = line_list[2].split(' ')
        time2index[index] = str(float("{:.2f}".format(float(value[0]))))
    if line_list[0] == 'EDGE_SE3_LOOP':
        value = line_list[1].split(' ')
        try:
          pose1 = frame_map[time2index[value[0]]]
          session1 = session_map[time2index[value[0]]]
          pose2 = frame_map[time2index[value[1]]]
          session2 = session_map[time2index[value[1]]]
          if session1 == session2:
              color = 'b'
          else:
              color=colors[(session1*10 + session2) % len(colors)]
          ax.plot([pose1[0], pose2[0]], [pose1[1], pose2[1]], [pose1[2], pose2[2]], color = color,linestyle='dashed', linewidth=1.5)
        except:
            continue   
    f.close
plt.title('3d trajectory view')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()

# 显示图形
plt.show()
