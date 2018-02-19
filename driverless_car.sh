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
  obstacle)
    cd /home/pi/repos/self_driving_car/utilities/
    sudo python3 obstacle_checker.py
    ;;
  capture)
    export CLOUDINARY_CLOUD_NAME="dtumd2ht6"
    export CLOUDINARY_API_KEY="849559948829221"
    export CLOUDINARY_API_SECRET="XgZ59pd95tt8QVnWfku36zNuMKg"
    export CLOUDINARY_URL=cloudinary://849559948829221:XgZ59pd95tt8QVnWfku36zNuMKg@dtumd2ht6
    cd /home/pi/repos/self_driving_car/utilities/
    sudo python3 capture_image.py
    ;;
  control)
    cd /home/pi/repos/self_driving_car/xbox_control/
    sudo python3 remote_control_car.py
    ;;
  stream)
    raspivid -o - -t 0 -vf -hf -fps 10 -b 500000 | ffmpeg -re -ar 44100 -ac 2 -acodec pcm_s16le -f s16le -ac 2 -i /dev/zero -f h264 -i - -vcodec copy -acodec aac -ab 128k -g 50 -strict experimental -f flv rtmp://a.rtmp.youtube.com/live2/su1r-euu7-g17y-69z4
    echo "Stream started"
    ;;
  *)
    # Refuse to do other stuff
    echo "Usage: /etc/init.d/testdaemon.sh {start|stop|restart}"
    exit 1
    ;;
esac

exit 0
