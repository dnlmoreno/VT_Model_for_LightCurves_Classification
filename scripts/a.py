from multiprocessing import Pool
import time

def test_process(x):
    print(f"Process {x} running", flush=True)
    time.sleep(180)
    return x

if __name__ == "__main__":
    num_workers = 30
    with Pool(processes=num_workers) as pool:
        results = pool.map(test_process, range(num_workers))
    print("Test completed:", results)