# standard python libs
import logging
import time
from neural_network_training.self_drive import StreamFrames

# third party libs
from daemon import runner


class DriverlessCarDaemon:
    def __init__(self):
        self.stdin_path = '/dev/null'
        self.stdout_path = '/dev/tty'
        self.stderr_path = '/dev/tty'
        self.pidfile_path = '/var/run/testdaemon/testdaemon.pid'
        self.pidfile_timeout = 5

    def run(self):
        while True:
            logger.info("Starting Vehicle")
            driver = StreamFrames()


app = DriverlessCarDaemon()
logger = logging.getLogger("DaemonLog")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.FileHandler("/var/log/testdaemon/testdaemon.log")
handler.setFormatter(formatter)
logger.addHandler(handler)

daemon_runner = runner.DaemonRunner(app)
# This ensures that the logger file handle does not get closed during daemonization
daemon_runner.daemon_context.files_preserve = [handler.stream]
daemon_runner.do_action()