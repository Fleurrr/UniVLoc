import torch
import multiprocessing
import time

def matrix_operation(device, matrix_size):
    torch.cuda.set_device(device)
    a = torch.rand(matrix_size, matrix_size).cuda()
    b = torch.rand(matrix_size, matrix_size).cuda()
    while True:
        result = torch.matmul(a, b)

        print("Matrix operation on GPU completed.")
        time.sleep(0.5)

if __name__ == "__main__":
    device_ids = [1]

    matrix_size = 30000    # 30000 in A100-40g

    with multiprocessing.Pool(processes=len(device_ids)) as pool:
        pool.starmap(matrix_operation, [(device, matrix_size) for device in device_ids])

