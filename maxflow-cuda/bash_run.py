import subprocess
import argparse


def execute_command(command):
    """Executes a single shell command and returns its output and error."""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

def batch_execute(commands, log_file_path):
    """Executes a list of commands in sequence and logs the output to a file."""
    with open(log_file_path, "w") as log_file:
        for command in commands:
            success, output = execute_command(command)
            if success:
                log_file.write(f"Command succeeded: {command}\nOutput:\n{output}\n")
            else:
                log_file.write(f"Command failed: {command}\nError:\n{output}\n")


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Execute a list of shell commands and log their output.")
    parser.add_argument("--log", help="Path to the log file.")
    args = parser.parse_args()


    # commands = [
    #     "./maxflow -v 2 -f dataset/DIMACC_format/Washington/Washington-RLG-small.txt -s 0 -t 0 -a 1",
    #     "./maxflow -v 2 -f dataset/DIMACC_format/Washington/Washington-RLG-small.txt -s 0 -t 0 -a 0"
    # ]
    # List of commands to execute
    commands = [
        "./maxflow -v 2 -f dataset/Segmentation/adhead.n6c10.max -s 0 -t 1 -a 0 -s 0 -t 1 -a 0",
        "./maxflow -v 2 -f dataset/Segmentation/adhead.n6c10.max -s 0 -t 1 -a 0 -s 0 -t 1 -a 1",
        "./maxflow -v 2 -f dataset/Segmentation/adhead.n26c100.max -s 0 -t 1 -a 0 -s 0 -t 1 -a 0",
        "./maxflow -v 2 -f dataset/Segmentation/adhead.n26c100.max -s 0 -t 1 -a 0 -s 0 -t 1 -a 1",
        # Add more commands as strings in this list
    ]

    batch_execute(commands, args.log)

