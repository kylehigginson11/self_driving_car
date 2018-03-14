#! /bin/bash

case "$1" in
  start)
    echo "Starting server"
    # Start the daemon
    cd /home/pi/repos/self_driving_car/neural_network_training/
    sudo python3 self_drive.py $2 $3 &
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
  obstacle)
    # Check for obstacle
    cd /home/pi/repos/self_driving_car/utilities/
    sudo python3 obstacle_checker.py
    ;;
  capture)
    # Capture image from camera
    export CLOUDINARY_CLOUD_NAME="dtumd2ht6"
    export CLOUDINARY_API_KEY="849559948829221"
    export CLOUDINARY_API_SECRET="XgZ59pd95tt8QVnWfku36zNuMKg"
    export CLOUDINARY_URL=cloudinary://849559948829221:XgZ59pd95tt8QVnWfku36zNuMKg@dtumd2ht6
    cd /home/pi/repos/self_driving_car/utilities/
    sudo python3 capture_image.py
    ;;
  control)
    # Control car manually with XBOX remote
    cd /home/pi/repos/self_driving_car/xbox_control/
    sudo python3 remote_control_car.py
    ;;
  collect)
    # collect training data
    cd /home/pi/repos/self_driving_car/neural_network_training/
    sudo python3 collect_training_images.py &
    ;;
  *)
    # Refuse to do other stuff
    echo "Usage: /etc/init.d/testdaemon.sh {start|stop|restart}"
    exit 1
    ;;
esac

exit 0
