import download_from_aip
import deal_download_data
import argparse
import os
import json
import math
import shutil
import numpy
import requests
import json
import os.path as osp
import os

if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    # 下载autoqa节点的结果 下载pn_mapping的输入 丛数据上看两个是同一个东西
    argsparse.add_argument('--deal_data', type=str, default="multi_mapping") # pn_mapping auto_qa  multi_mapping
    argsparse.add_argument('--deal_mode', type=str, default="looplose") # result_single, result_multi, process(暂时未有需求), looplose floorchange
    argsparse.add_argument('--aip_link_path', type=str, default="./aip_links.txt")    
    argsparse.add_argument('--root_dir', type=str, default="/map-hl/gary.huang/training_2025")

    argsparse.add_argument('--plus_img_filter', type=bool, default=False) # 是否额外下载imgfilter的内容
    args = argsparse.parse_args()

    aip_links = []
    with open(args.aip_link_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            aip_links.append(line.strip())

    print("start to download data")
    data_save_root = os.path.join(args.root_dir, args.deal_data)
    # data_save_root = args.root_dir
    if not os.path.exists(data_save_root):
        os.mkdir(data_save_root)
    
    plus_img_filter = args.plus_img_filter
    for aip_link in aip_links:
      map_id = download_from_aip.download_pn_mapping_batch(aip_link, data_save_root, args.deal_data, plus_img_filter)
      print("data_save_path", data_save_root , "map_id",map_id)
      data_save_path = os.path.join(data_save_root, map_id)
      print("start to deal data")
      deal_download_data.process_directory(data_save_path, args.deal_data)