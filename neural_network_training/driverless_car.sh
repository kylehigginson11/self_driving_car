#! /bin/bash

case "$1" in
  start)
    echo "Starting server"
    # Start the daemon
    cd /home/pi/repos/self_driving_car/neural_network_training/
    sudo python3 self_drive.py &
    ;;
  stop)
    echo "Stopping server"
    # Stop the daemon
    sudo pkill python
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
