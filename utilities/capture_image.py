import picamera
import time
import sys
import os
import random
import string
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url
from cloudinary.api import delete_resources_by_tag, resources_by_tag
import cloudinary

cloudinary.config(
    cloud_name='kylehigginson',
    api_key='315568387175561',
    api_secret='-ZV4Khu3SCZVT4P5EYq4_cmqWVM'
)

DEFAULT_TAG = "albert_photo"


def upload_files(public_id):
    # print("--- Upload a local file")
    response = upload("upload.jpg", tags=DEFAULT_TAG, public_id=public_id + "!")
    url, options = cloudinary_url(response['public_id'],
                                  format=response['format'],
                                  crop="fill"
                                  )
    # print("Fill 200x150 url: " + url)
    # print("")
    sys.stdout.write(str(url))
    sys.stdout.flush()


def cleanup_cloudinary():
    response = resources_by_tag(DEFAULT_TAG)
    resources = response.get('resources', [])
    if not resources:
        # print("No images found")
        return
    # print("Deleting {0:d} images...".format(len(resources)))
    delete_resources_by_tag(DEFAULT_TAG)
    # print("Done!")
    pass


def take_picture(public_id):
    # print ("Capturing image, say cheese!")
    camera = picamera.PiCamera()
    camera.rotation = 180
    camera.capture('upload.jpg')
    time.sleep(1)
    camera.close()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'upload': upload_files(public_id)
        if sys.argv[1] == 'cleanup': cleanup_cloudinary()
    else:
        # print("--- Uploading files and then cleaning up")
        # print("    you can only one instead by passing 'upload' or 'cleanup' as an argument")
        # print("")
        upload_files(public_id)
        os.remove("upload.jpg")


def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


public_name = random_generator()
data = public_name + "!"
take_picture(public_name)
