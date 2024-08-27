import subprocess
import argparse
import os
import sys
import re
import numpy as np

def parse_breakdown_times(output):
    execution_times = {
        "Kernel execution time": f"{float(re.search(r'Kernel execution time:\s+([\d\.]+)\s+ms', output).group(1)):.2f} ms",
        "Neighbor searching time": f"{float(re.search(r'Neighbor searching time:\s+([\d\.]+)\s+ms', output).group(1)):.2f} ms",
        "Backward finding time": f"{float(re.search(r'Backward finding time:\s+([\d\.]+)\s+ms', output).group(1)):.2f} ms",
        "Other time": f"{float(re.search(r'Other time:\s+([\d\.]+)\s+ms', output).group(1)):.2f} ms",
    }
    return execution_times


def execute_command(command):
    """Executes a single shell command and returns its output and error."""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

def batch_execute(commands, log_file, times_file, stats_file=None, breakdown_file=None):
    """Executes a list of commands in sequence, logs the output, and captures execution times."""
    time_regex = re.compile(r"Total kernel time: ([\d.]+) ms")

    # Regular expressions for parsing the output of the workload analysis
    warps_regex = re.compile(r"#warps: (\d+)")
    warps_time_regex = re.compile(r"Warp execution time:\n([\d\s.]+)")

    for command in commands:
        success, output = execute_command(command)
        if success:
            log_file.write(f"Command succeeded: {command}\nOutput:\n{output}\n")
            # Search for the execution time in the output
            match = time_regex.search(output)
            if match:
                print("Match found")
                execution_time = match.group(1)
                times_file.write(f"{command}: {execution_time} ms\n")

            # Extract and process warp execution times
            if stats_file is not None:
                warps_match = warps_regex.search(output)
                warps_time_match = warps_time_regex.search(output)
                if warps_match and warps_time_match:
                    num_warps = int(warps_match.group(1))
                    warps_times = list(map(float, warps_time_match.group(1).split()))
                    if len(warps_times) == num_warps:
                        # Calculate statistics
                        min_time = np.min(warps_times)
                        lower_quartile = np.percentile(warps_times, 25)
                        median = np.median(warps_times)
                        upper_quartile = np.percentile(warps_times, 75)
                        max_time = np.max(warps_times)
                        avg_time = np.mean(warps_times)
                        stats = f"Min: {min_time}, Lower Quartile: {lower_quartile}, Median: {median}, Upper Quartile: {upper_quartile}, Max: {max_time}, Avg: {avg_time}"
                        stats_file.write(f"{command}:\n\t{stats}\n")

            # Extract and process execution time breakdown
            if breakdown_file is not None:
                execution_times = parse_breakdown_times(output)
                breakdown_file.write(f"{command}:\n")
                for key, value in execution_times.items():
                    breakdown_file.write(f"\t{key}: {value}\n")
                    
        else:
            log_file.write(f"Command failed: {command}\nError:\n{output}\n")


def generate_commands(directory):
    """Generates a list of commands based on files in a given directory."""
    commands = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            command = f"./maxflow -v 2 -s 0 -t 1 -f {filepath} -a 0"
            commands.append(command)
            command = f"./maxflow -v 2 -s 0 -t 1 -f {filepath} -a 1"
            commands.append(command)
    return commands


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Execute a list of shell commands, log their output, and capture execution times.")
    parser.add_argument("--log", help="Path to the log file or 'stdout' for console output.", default="stdout")
    parser.add_argument("--dir", help="Path to the directory where the input files to execute.", required=True)
    parser.add_argument("--times", help="Path to the file where execution times will be logged.", required=True)
    parser.add_argument("--stats", help="Path to the file where execution time statistics will be logged.", default=None)
    parser.add_argument("--breakdown", help="Path to the file where execution time breakdown will be logged.", default=None)
    args = parser.parse_args()

    # Open log files
    if args.log == "stdout":
        log_file = sys.stdout
    else:
        log_file = open(args.log, "w")


    times_file = open(args.times, "w")

    if args.stats is not None:
        stats_file = open(args.stats, "w")
    else:
        stats_file = None

    if args.breakdown is not None:
        breakdown_file = open(args.breakdown, "w")
    else:
        breakdown_file = None

    commands = generate_commands(args.dir)

    batch_execute(commands, log_file, times_file, stats_file, breakdown_file)

    if args.log != "stdout":
        log_file.close()
    times_file.close()
    if stats_file is not None:
        stats_file.close()
