#!/bin/bash

# v4l2-ctl -d /dev/video2 --list-ctrls -> Allows to list the available controls

# User Controls:

#                      brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=0
#                        contrast 0x00980901 (int)    : min=0 max=255 step=1 default=32 value=32
#                      saturation 0x00980902 (int)    : min=0 max=255 step=1 default=32 value=32
#         white_balance_automatic 0x0098090c (bool)   : default=1 value=1
#                            gain 0x00980913 (int)    : min=0 max=255 step=1 default=64 value=192
#            power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2
#       white_balance_temperature 0x0098091a (int)    : min=0 max=10000 step=10 default=4000 value=5070 flags=inactive
#                       sharpness 0x0098091b (int)    : min=0 max=255 step=1 default=24 value=24
#          backlight_compensation 0x0098091c (int)    : min=0 max=1 step=1 default=0 value=0

# Camera Controls:

#                   auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
#          exposure_time_absolute 0x009a0902 (int)    : min=1 max=10000 step=1 default=166 value=336 flags=inactive
#      exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1


v4l2-ctl -d /dev/video2 --set-ctrl=white_balance_automatic=1
v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=0
v4l2-ctl -d /dev/video2 --set-ctrl=backlight_compensation=1
v4l2-ctl -d /dev/video2 --set-ctrl=contrast=200
v4l2-ctl -d /dev/video2 --set-ctrl=saturation=170  ## Crucial 
v4l2-ctl -d /dev/video2 --set-ctrl=gain=0 ##
v4l2-ctl -d /dev/video2 --set-ctrl=white_balance_temperature=4000   #Interesing behavior at 4000
v4l2-ctl -d /dev/video2 --set-ctrl=sharpness=128
v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute=250
v4l2-ctl -d /dev/video2 --set-ctrl=brightness=100 ##|

