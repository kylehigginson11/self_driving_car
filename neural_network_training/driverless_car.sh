#! /bin/bash
# Copyright (c) 1996-2012 My Company.
# All rights reserved.
#
# /etc/init.d/driverless_car
#
### BEGIN INIT INFO
# Provides: testdaemon
# Required-Start:
# Should-Start:
# Required-Stop:
# Should-Stop:
# Default-Start:  3 5
# Default-Stop:   0 1 2 6
# Short-Description: Test daemon process
# Description:    Runs up the test daemon process
### END INIT INFO

# Activate the python virtual environment

case "$1" in
  start)
    echo "Starting server"
    # Start the daemon
    cd ~/repos/self_driving_car/neural_network_training/
    sudo python3 self_drive.py &
    LASTPID=$!
    ;;
  stop)
    echo "Stopping server"
    # Stop the daemon
    kill $LASTPID
    ;;
  restart)
    echo "Restarting server"
    # python /usr/share/testdaemon/driverless_car_daemon.py restart
    ;;
  *)
    # Refuse to do other stuff
    echo "Usage: /etc/init.d/testdaemon.sh {start|stop|restart}"
    exit 1
    ;;
esac

exit 0