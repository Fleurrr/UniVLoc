'''
Author: yongnan.chen yongnan.chen@nio.com
Date: 2024-09-11 10:49:32
LastEditors: yongnan.chen yongnan.chen@nio.com
LastEditTime: 2024-11-07 19:44:54
FilePath: /map_sdk/home/nio/hdd/datasets/get_semantic_geojson_from_aip_task_url.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

"""Get bucket path from aip task url.

"""
import os
import json
import math

import numpy
import requests
import json
import requests
import tarfile
import shutil

import argparse

TASK_URL_DEV = "http://data-processing-task-management.tencent-dev.nioint.com/"
TASK_URL_PROD = "http://data-processing-task-management.tencent-prod.nioint.com/"
JOB_URL_DEV = "http://data-processing-workflow-management.tencent-dev.nioint.com/"
JOB_URL_PROD = "http://data-processing-workflow-management.tencent-prod.nioint.com/"
niofs_bin = '/home/g.huang/niofs-linux/bin/niofs'# '/home/nio/niofs-linux/bin/niofs'
MS_TO_MIN = 60000
download_node = ""

def NIOFSCopy(bin, link, target_folder):
    cmd = '{0} copy {1} {2} -P'.format(bin, link, target_folder)
    os.system(cmd)

def get_task_instances(task_id, env="prod", limit=10) -> dict:
    if env == "dev":
        base_url = TASK_URL_DEV + f"api/workflow/tasks/{task_id}/list"
    elif env == "prod":
        base_url = TASK_URL_PROD + f"api/workflow/tasks/{task_id}/list"
    else:
        print("abnormal env")
        return {}

    job_instances = {}
    offset = 0

    while True:
        remote_path = f"{base_url}?limit={limit}&offset={offset}"
        response = requests.get(remote_path).json()
        task_instances = response.get("workflow_instances", [])
        if not task_instances:
            break  # 如果没有更多数据，结束循环
        for instance in task_instances:
            assert instance["task_id"] == task_id
            # print("instance: ", instance["id"])
            # if(instance["id"] != 1033170697):
                # continue
            # TODO: 需要只统计成功数据时打开这个注释
            if instance["status"] != "SUCCEEDED":
                continue
            job_instances[instance["id"]] = {
                "task_all_time": (instance["finished_at"] - instance["started_at"]) / MS_TO_MIN
            }
        offset += limit

    return job_instances

def get_node_input(job_resp):
    job_task = job_resp["task"]
    job_proc_kwargs = job_task["proc_kwargs"]
    data = json.loads(job_proc_kwargs)
    map_id_value = data.get("map_id", None)
    job_result = job_task["composite_clip_data"]
    processed_clip_path = job_result["clip_data"][0]["processed_clip_data"]["processed_clip_path"]
    return (map_id_value,processed_clip_path)

def get_node_output(job_resp):
    job_task = job_resp["task"]
    job_proc_kwargs = job_task["proc_kwargs"]
    data = json.loads(job_proc_kwargs)
    map_id_value = data.get("map_id", None)
    job_result = job_resp["result"]
    processed_clip_path = job_result["succeeded_message"]["composite_data"]["clip_data"][0]["processed_clip_data"]["processed_clip_path"]
    return (map_id_value,processed_clip_path)

def get_task_execution(task_id, job_id, plus_img_filter, env="prod") -> dict:
    if env == "dev":
        base_url  = JOB_URL_DEV + f"task_executions?workflow_task_id={task_id}&workflow_job_id={job_id}"
    elif env == "prod":
        base_url  = JOB_URL_PROD + f"task_executions?workflow_task_id={task_id}&workflow_job_id={job_id}"
    else:
        print("abnormal env")
        return {}
    task_executions = []
    offset = 0
    limit = 10
    while True:
        remote_path = f"{base_url}&offset={offset}&limit={limit}"
        job_resp = requests.get(remote_path).json()
        if "task_execution" not in job_resp or not job_resp["task_execution"]:
            break
        task_executions.extend(job_resp["task_execution"])
        if len(job_resp["task_execution"]) < limit:
            break
        offset += limit
    job_infos = {"job_num": len(task_executions)}
    print("download node:", job_infos)
    if len(task_executions) < 2:
        return job_infos
    
    clip_info_pair = []
    for job_resp in task_executions:
        if download_node == "pn_mapping" and job_resp["app_id"] == "k8s_job:adda-pn-mapping-pn-mapping":
            print("get pn mapping")
            clip_info_pair.append(get_node_output(job_resp))
        elif download_node == "auto_qa" and job_resp["app_id"] == "k8s_job:adda-pn-mapping-qa":
            print("get auto qa")
            clip_info_pair.append(get_node_input(job_resp))
        elif download_node == "multi_mapping" and job_resp["app_id"] == "k8s_job:adda-pn-mapping-multi-session-fusion":
            print("get multi mapping")
            clip_info_pair.append(get_node_output(job_resp))        
    
    #print("plus_img_filter:", plus_img_filter)
    for job_resp in task_executions:
        if plus_img_filter and job_resp["app_id"] == "k8s_job:adda-pn-mapping-image-filter" and job_resp["workflow_step_task"]["workflow_step_id"] == "33":
            #print(job_resp)
            print("get fliter session")
            clip_info_pair.append(get_node_output(job_resp))

    return clip_info_pair

def GetSessionNames(maps_save_folder):
    sessions = []
    map_json_file = os.path.join(maps_save_folder, 'map.json')
    with open(map_json_file, 'r') as file: 
        json_data = json.load(file)
        sessions.append(json_data['session_info']['base_session'])
        if 'new_session' in json_data['session_info']:
            for session in json_data['session_info']['new_session']:
                sessions.append(session)
    return sessions

def download_pn_mapping(pn_result_qcloud, data_save_folder):
    map_bin = pn_result_qcloud + '/maps/map.json'
    maps_save_folder = data_save_folder + '/maps'
    others_save_folder = data_save_folder + '/others'
    sessions_save_folder = data_save_folder + '/sessions'
    if not os.path.exists(data_save_folder):
        os.mkdir(data_save_folder)
    other_tar = pn_result_qcloud + "/others.tar"
    sessions_tar = pn_result_qcloud + "/sessions.tar"
    map_tar = pn_result_qcloud + "/map.tar"
    if_download_pn_result = True
    if not os.path.exists(data_save_folder):
        os.mkdir(data_save_folder)
    NIOFSCopy(bin=niofs_bin, link=map_bin, target_folder=maps_save_folder)
    NIOFSCopy(bin=niofs_bin, link=other_tar, target_folder=data_save_folder)
    NIOFSCopy(bin=niofs_bin, link=sessions_tar, target_folder=data_save_folder)
    sessions = GetSessionNames(maps_save_folder)
    for session in sessions:
        session_other = pn_result_qcloud + "/others/" + session
        NIOFSCopy(bin=niofs_bin, link=session_other, target_folder=others_save_folder + "/" + session)
    NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/others/global_grid_mapping_result/global_semantic_grid_map.pcd", target_folder=others_save_folder + "/global_grid_mapping_result")
    NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/others/intersection_detect_result/intersection.json", target_folder=others_save_folder + "/intersection_detect_result")
    NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/others/semantic_result/semantic_vector.pb.bin", target_folder=others_save_folder + "/semantic_result")
    NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/others/sessions/map.json", target_folder=others_save_folder)
    NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/others/fusion_results/map.json", target_folder=others_save_folder + "/fusion_results")
    NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/sessions/map.json", target_folder=sessions_save_folder)
    if if_download_pn_result:
        NIOFSCopy(bin=niofs_bin, link=pn_result_qcloud + "/others/pn_mapping", target_folder=others_save_folder + "/pn_mapping")

def download_img_filter(pn_result_qcloud, data_save_folder):
    print("download img filter:", data_save_folder)
    if not os.path.exists(data_save_folder):
        os.mkdir(data_save_folder)
    map_tar = pn_result_qcloud + "/map.tar"
    NIOFSCopy(bin=niofs_bin, link=map_tar, target_folder=data_save_folder)


def download_multi_mapping(pn_result_qcloud, data_save_folder):
    print("download multi mapping:", data_save_folder)
    if not os.path.exists(data_save_folder):
        os.mkdir(data_save_folder)
    #map_tar = pn_result_qcloud + "/map.tar"
    session_tar = pn_result_qcloud + "/sessions.tar"
    debug_tar = pn_result_qcloud + "/debug.tar"
    map_tar = pn_result_qcloud + "/map.tar"
    #NIOFSCopy(bin=niofs_bin, link=map_tar, target_folder=data_save_folder)    
    NIOFSCopy(bin=niofs_bin, link=session_tar, target_folder=data_save_folder) 
    # NIOFSCopy(bin=niofs_bin, link=debug_tar, target_folder=data_save_folder) 
    NIOFSCopy(bin=niofs_bin, link=map_tar, target_folder=data_save_folder) 


def download_pn_mapping_batch(aip_link, data_save_folder, download_data, plus_img_filter):
    global download_node
    download_node = download_data
    #print("data_save_folder :", data_save_folder)
    if not os.path.exists(data_save_folder):
        os.mkdir(data_save_folder)
    task_url = aip_link
    #  = task_url.split('&')[-1].split('=')[-1]
    # print("select_instance_id:",int(select_instance_id))
    task_id = int(task_url[task_url.rfind("/")+1:task_url.rfind("?")])
    instances = get_task_instances(task_id, env="prod")    # get all instance of task
    #print("instances:", instances)
    map_id = 0
    for instance in instances:
        # if instance != int(select_instance_id):
        #     continue
        result_list = get_task_execution(task_id, instance, plus_img_filter, env="prod")
        print("len result list:", len(result_list))
        if result_list is not None:
            if result_list[0] is not None:
                map_id, processed_clip_path = result_list[0]
                if map_id is not None and processed_clip_path is not None:
                    print(f"map_id: {map_id}, processed_clip_path: {processed_clip_path}")
                    # niofs copy
                    target_folder=data_save_folder + "/" + map_id
                    if os.path.exists(target_folder):
                        print("already have case:", map_id, " continue")
                        continue
                    if not data_save_folder.endswith("multi_mapping"):
                        print("download pn mapping")
                        download_pn_mapping(processed_clip_path, target_folder)
                    else:
                        print("download multi mapping")
                        download_multi_mapping(processed_clip_path, target_folder)
            else:
                print(f"task_id: {task_id} get None Pn-mapping Result")

            if plus_img_filter and len(result_list) == 2:
                print("plus download img filter")
                if result_list[1] is not None:
                    map_id, processed_clip_path = result_list[1]
                    if map_id is not None and processed_clip_path is not None:
                        print(f"download img filter  map_id: {map_id}, processed_clip_path: {processed_clip_path}")
                        # niofs copy
                        target_folder=data_save_folder + "/" + map_id + "/img_filter"

                        if os.path.exists(target_folder):
                            print("already have case:", map_id, " continue")
                            continue
                        download_img_filter(processed_clip_path, target_folder)
    return map_id

if __name__ == "__main__":
    task_url_batch = ["https://aip.nioint.com/#/general-computing/workflow/task/list/detail/118939686?tab=job&instance_id=1040492231"]
    data_save_folder = '/home/nio/data/adviz_data_1108/tmp_pn_mapping'
    if not os.path.exists(data_save_folder):
        os.mkdir(data_save_folder)
    for task_url in task_url_batch:
        select_instance_id = task_url.split('&')[-1].split('=')[-1]
        print("select_instance_id:",int(select_instance_id))
        task_id = int(task_url[task_url.rfind("/")+1:task_url.rfind("?")])
        instances = get_task_instances(task_id, env="prod")    # get all instance of task
        for instance in instances:
            if instance != int(select_instance_id):
                continue
            result = get_task_execution(task_id, instance, env="prod")
            if result is not None:
                map_id, processed_clip_path = result
                if map_id is not None and processed_clip_path is not None:
                    print(f"map_id: {map_id}, processed_clip_path: {processed_clip_path}")
                    # niofs copy
                    target_folder=data_save_folder + "/" + map_id
                    download_pn_mapping(processed_clip_path, target_folder)
            else:
                print(f"task_id: {task_id} get None Pn-mapping Result")
        