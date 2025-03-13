import os
import subprocess

def get_cpu_info():
    cpu_info = {}
    lscpu_output = subprocess.check_output("lscpu", shell=True).decode()
    for line in lscpu_output.split("\n"):
        if "Model name:" in line:
            cpu_info["Model"] = line.split(":")[1].strip()
        if "CPU(s):" in line and "NUMA" not in line:
            cpu_info["Cores"] = line.split(":")[1].strip()
        if "Thread(s) per core:" in line:
            cpu_info["Threads per core"] = line.split(":")[1].strip()
        if "CPU MHz:" in line:
            cpu_info["Base Clock"] = line.split(":")[1].strip() + " MHz"
        if "CPU max MHz:" in line:
            cpu_info["Boost Clock"] = line.split(":")[1].strip() + " MHz"
    return cpu_info

def get_gpu_info():
    gpu_info = {}
    try:
        nvidia_smi_output = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True).decode()
        # print (nvidia_smi_output)
        gpu_models = set()
        gpu_vrams = set()
        for line in nvidia_smi_output.strip().split("\n"):
            model, vram = [value.strip() for value in line.split(",")]
            gpu_models.add(model)
            gpu_vrams.add(vram)
        gpu_info["Model"] = ", ".join(gpu_models)
        gpu_info["VRAM"] = ", ".join(gpu_vrams)
    except subprocess.CalledProcessError:
        gpu_info["Model"] = "N/A"
        gpu_info["VRAM"] = "N/A"
    return gpu_info

def get_ram_info():
    ram_info = {}
    meminfo_output = subprocess.check_output("free -h", shell=True).decode()
    for line in meminfo_output.split("\n"):
        if "Mem:" in line:
            ram_info["Total"] = line.split()[1]
    return ram_info

def get_storage_info():
    storage_info = {}
    lsblk_output = subprocess.check_output("lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep 'disk'", shell=True).decode()
    for line in lsblk_output.split("\n"):
        if line:
            name, size, type = line.split()
            storage_info[name] = size
    return storage_info

def main():
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    ram_info = get_ram_info()
    storage_info = get_storage_info()

    print("### Hardware Specifications:")
    print(f"- **CPU**: {cpu_info.get('Model', 'N/A')}, {cpu_info.get('Cores', 'N/A')} cores, {cpu_info.get('Threads per core', 'N/A')} threads, {cpu_info.get('Base Clock', 'N/A')} base clock, {cpu_info.get('Boost Clock', 'N/A')} boost clock")
    print(f"- **GPU**: {gpu_info.get('Model', 'N/A')}, {gpu_info.get('VRAM', 'N/A')}")
    print(f"- **RAM**: {ram_info.get('Total', 'N/A')}")
    print(f"- **Storage**: {', '.join([f'{name}: {size}' for name, size in storage_info.items()])}")

if __name__ == "__main__":
    main()