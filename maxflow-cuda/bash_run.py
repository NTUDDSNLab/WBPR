import subprocess
import argparse
import os
import sys
import re

def execute_command(command):
    """Executes a single shell command and returns its output and error."""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

def batch_execute(commands, log_file, times_file):
    """Executes a list of commands in sequence, logs the output, and captures execution times."""
    time_regex = re.compile(r"Total kernel time: ([\d.]+) ms")
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
    args = parser.parse_args()

    if args.log == "stdout":
        log_file = sys.stdout
    else:
        log_file = open(args.log, "w")

    times_file = open(args.times, "w")

    commands = generate_commands(args.dir)

    batch_execute(commands, log_file, times_file)

    if args.log != "stdout":
        log_file.close()
    times_file.close()
