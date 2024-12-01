import psutil
import sys
import time
import json

# class Logger(object):
#   def __init__(self, path):
#     self.terminal = sys.stdout
#     self.log = open(path, "w")

#   def write(self, message):
#     # self.terminal.write(message)
#     self.log.write(message)

#   def flush(self):
#     # self.terminal.flush()
#     self.log.flush()

processes = []
process_names = []

def get_process():
  global processes
  global process_names
  # for i in range(0, len(processes)):
  #   if processes[i].is_running() == False:
  #     print(f'process not found: {process_names[i]}')
  processes = []
  process_names = []
  for proc in psutil.process_iter():
    if 'llama-server' in proc.name():
      # print(proc.pid)
      # print(proc.name)
      processes.append(proc)
      process_names.append(proc.name())

def do(i, file_name):
    global processes
    global process_names
    exception = True
    data = {"time": i}
    retries = 0  # Retry count for exception handling

    while exception and retries < 7:
        try:
            get_process()
            for name in process_names:
                data[name] = {"cpu": 0.0, "mem": 0.0}
            for proc in processes:
                try:
                    with proc.oneshot():
                        cpu = proc.cpu_percent(interval=None)
                        mem = proc.memory_info().rss
                    data[proc.name()]['cpu'] = cpu
                    data[proc.name()]['mem'] = mem
                except psutil.NoSuchProcess:
                    print(f"Process terminated: {proc.name()}")
                    continue
        except Exception as error:
            print(f"An error occurred: {error}")
            retries += 1
        else:
            exception = False

    if not exception:
        try:
            with open(file_name, "a") as file:
                json.dump(data, file)
                file.write('\n')
                file.flush()
        except Exception as error:
            print(f"Error writing to file: {error}")

if __name__ == '__main__':
    time_interval = 1.0
    start_time = int(sys.argv[2:][0])
  # start_time = (time.time() + 5) * 1e9
    print(f'start time: {start_time}')
    print(f'log file: {sys.argv[1]}')
  

    i = -1
    time_delta = start_time / 1e9 - time.time()
    time.sleep(time_delta if time_delta > 0 else 0)
    with open(sys.argv[1], "w") as file:
        file.write(f'start time: {start_time}\n')
    while 1:
        do(i, sys.argv[1])
        i = i + 1
        time.sleep(1)