import os
import tarfile
import shutil

def pretreat_pnmapping_data(directory):
    dealed_case = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item == 'others.tar':
            dealed_case = dealed_case + 1
        if os.path.isfile(item_path) and item == 'sessions.tar':
            dealed_case = dealed_case + 1
    if dealed_case < 2 :
        print("curr case has been extract, continue")
        return
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        print("item path:", item_path)
        if os.path.isdir(item_path) and item == "img_filter":
            img_filter_map_path = os.path.join(item_path, 'map.tar')
            with tarfile.open(img_filter_map_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'img_filter/map'))             
            os.remove(item_path)
            print(f"Deleted '{item_path}'")

        # 检查是否是名为'others'的文件夹
        if os.path.isdir(item_path) and item == 'others':
            # 重命名文件夹为'others_info'
            new_path = os.path.join(directory, 'others_info')
            if os.path.exists(new_path):
                continue
            os.rename(item_path, new_path)
            print(f"Renamed '{item_path}' to '{new_path}'")
        
        # 检查是否是名为'sessions'的文件夹
        elif os.path.isdir(item_path) and item == 'sessions':
            # 重命名文件夹为'sessions_info'
            new_path = os.path.join(directory, 'sessions_info')
            if os.path.exists(new_path):
                continue
            os.rename(item_path, new_path)
            print(f"Renamed '{item_path}' to '{new_path}'")
        
        # 检查是否是名为'others.tar'的文件
        elif os.path.isfile(item_path) and item == 'others.tar':
            # 解压文件到'others'文件夹
            with tarfile.open(item_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'others_data'))
            print(f"Extracted '{item_path}' to 'others_data' folder")
            os.remove(item_path)
            print(f"Deleted '{item_path}'")

        # 检查是否是名为'sessions.tar'的文件
        elif os.path.isfile(item_path) and item == 'sessions.tar':
            # 解压文件到'sessions'文件夹
            with tarfile.open(item_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'sessions_data'))
            print(f"Extracted '{item_path}' to 'sessions_data' folder")
            os.remove(item_path)
            print(f"Deleted '{item_path}'")

    # 把others_info文件夹里的 global_grid_mapping_result 的 global_semantic_grid_map.pcd 
    #  复制到 others_data的  global_grid_mapping_result 里

    source_file = os.path.join(directory, 'others_info/global_grid_mapping_result/global_semantic_grid_map.pcd')
    destination_folder = os.path.join(directory, 'others_data/global_grid_mapping_result') 
    # 复制文件
    shutil.copy(source_file, destination_folder)

def pretreat_multi_mapping(directory):
    dealed_case = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and (item == 'map.tar' or item == 'debug.tar'):
            dealed_case = dealed_case + 1
        if os.path.isfile(item_path) and item == 'sessions.tar':
            dealed_case = dealed_case + 1    
    if dealed_case < 2 :
        print("curr case has been extract, continue")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        print("item path:", item_path)
        if os.path.isdir(item_path) and item == "img_filter":
            img_filter_map_path = os.path.join(item_path, 'map.tar')
            with tarfile.open(img_filter_map_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'img_filter/map'))
            os.remove(img_filter_map_path)
            print(f"Deleted '{img_filter_map_path}'")             

        # 检查是否是名为'others.tar'的文件
        elif os.path.isfile(item_path) and item == 'map.tar':
            # 解压文件到'others'文件夹
            with tarfile.open(item_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'map_data'))
            print(f"Extracted '{item_path}' to 'map_data' folder")
            os.remove(item_path)
            print(f"Deleted '{item_path}'")

        # 检查是否是名为'sessions.tar'的文件
        elif os.path.isfile(item_path) and item == 'sessions.tar':
            # 解压文件到'sessions'文件夹
            with tarfile.open(item_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'sessions_data'))
            print(f"Extracted '{item_path}' to 'sessions_data' folder")
            os.remove(item_path)
            print(f"Deleted '{item_path}'")

        elif os.path.isfile(item_path) and item == 'debug.tar':
            # 解压文件到'sessions'文件夹
            with tarfile.open(item_path, 'r:*') as tar:
                tar.extractall(path=os.path.join(directory, 'debug_data'))
            print(f"Extracted '{item_path}' to 'debug_data' folder")
            os.remove(item_path)
            print(f"Deleted '{item_path}'")

def process_directory(directory, deal_data):
    # 遍历当前目录下的所有第一层子目录和文件
    if deal_data == "pn_mapping" or deal_data == "auto_qa":
        pretreat_pnmapping_data(directory)
    elif deal_data == "multi_mapping":
        pretreat_multi_mapping(directory)


if __name__ == "__main__":
    # 替换下面的路径为你想要处理的目录路径
    target_directory = "/home/nio/data/adviz_data_1108/tmp_pn_mapping/PK-Downtown-93f06f2a-fe7d7757"
    process_directory(target_directory)