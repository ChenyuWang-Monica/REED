# This script serves like a daemon that keeps the drug pcdm training: When it tears down, it automatically re-run it with the latest checkpoint.

import hydra
import logging
import subprocess
import os, sys
import datetime
import time
logger = logging.getLogger(__name__)

def get_latest_modification_time(path):
    if os.path.isdir(path):
        # If it's a directory, find the latest modification time among all files
        latest_modification_time = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_mod_time = os.path.getmtime(file_path)
                if file_mod_time > latest_modification_time:
                    latest_modification_time = file_mod_time
        modification_time = latest_modification_time
    else:
        # If it's a file, get its modification time
        modification_time = os.path.getmtime(path)

    formatted_time = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

@hydra.main(config_path="../configs", config_name="monitor.yaml", version_base="1.3")
def monitor(cfg):
    monitor_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Monitor started at: {monitor_start_time}")
    logger.info(f"Monitor Args: {cfg}")
    logger.info(f"Current Working Dir: {os.getcwd()}")
    logger.info(f"Current Python executable: {sys.executable}")
    
    # IAI Configs
    if cfg.IAI:
        logger.info("Running on IAI, setting up environment...")

        # Export environment variables for WANDB and OMP threads
        os.environ["WANDB_API_KEY"] = "96f21629a21b7d93ef6ea6a2c57466bc562414b9" # Zian's wandb
        os.environ["OMP_NUM_THREADS"] = "4"
        
        logger.info(f"WANDB_API_KEY: {os.environ['WANDB_API_KEY']}")
        logger.info(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
        
        # Build the srun command
        srun_prefix = [
            "srun",
            f"--job-name={cfg.IAI_job_name}",
            f"--partition={cfg.partition}",
            f"--nodelist={cfg.nodelist}",
            f"--qos={cfg.hgx}",
            f"--time={cfg.time}",
            "--nodes=1",
            f"--gres={cfg.gres}",
            "-c", "12"
        ]
        
        logger.info(f"Srun Arguments: {srun_prefix}")



    # Run subprocess
    subprocess_args = cfg.subprocess_args
    subprocess_args = subprocess_args.split(" ")
    logger.info(f"Subprocess Hydra Arguments: {subprocess_args}")
        
    version = next((arg.split("=")[1] for arg in subprocess_args if arg.startswith(f"{cfg.version_arg_name}=")), None)
    assert version is not None, f"Please ensure that you have specified {cfg.version_arg_name} externally so that the monitor can track the process!"
    logger.info(f"Extracted version: {version}")
    
    def start_process(subprocess_args): 
        if cfg.DDP:
            Popen_args = ['python', '-m', 'torch.distributed.run', f"--master_port={cfg.DDP_port}", f"--nproc_per_node={cfg.DDP_nproc_per_node}", f'src/{cfg.script_name}.py']
        else:
            Popen_args = ['python', f'src/{cfg.script_name}.py']
        Popen_args.extend(subprocess_args)
        if cfg.IAI:
            Popen_args = srun_prefix + Popen_args
        return subprocess.Popen(Popen_args)
    process = start_process(subprocess_args)
    logger.info(f"Started process with PID: {process.pid}")
    

    
    # Monitor subprocess
    try:
        while True:
            # Check if the process has terminated
            if process.poll() is not None:  # process.poll() returns None if it's still running
                logger.info(f"Process {process.pid} terminated. Restarting...")
                # Check if there is a checkpoint.
                if cfg.last_ckpt_name is None:
                    path = f"{cfg.output_dir}/{version}"
                else:
                    path = f"{cfg.output_dir}/{version}/{cfg.last_ckpt_name}"
                assert os.path.exists(path), f"The last checkpoint {path} does not exist!"
                
                
                formatted_time = get_latest_modification_time(path)
                logger.info(f"Last checkpoint {path} was modified at: {formatted_time}")
                
                index = next((i for i, arg in enumerate(subprocess_args) if arg.startswith(cfg.resume_arg_name)), None)
                if index is not None:
                    subprocess_args[index] = f"{cfg.resume_arg_name}={path}"
                else:
                    subprocess_args.extend([f"{cfg.resume_arg_name}={path}"])
                
                process = start_process(subprocess_args)  # Restart the process
                logger.info(f"Restarted process with PID: {process.pid}")
                logger.info(f"Restarted process with subprocess_args: {subprocess_args}")
                
            
            # Add a sleep interval to avoid tight looping and CPU consumption
            time.sleep(int(cfg.monitor_interval))
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
        process.terminate()
        try:
            process.wait(timeout=10)  # Ensure graceful shutdown with a timeout
        except subprocess.TimeoutExpired:
            process.kill()  # Forcefully kill if it's not shutting down
            logger.warning(f"Process {process.pid} killed due to timeout.")
        
if __name__ == "__main__":
    monitor()