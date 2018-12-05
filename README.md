# FinalCode
Final Project Submission

To run SNN.py, below are the dependicies you have to full fill:
1) Mention dependencies_loc as location where you kept below files:
      - pose_deploy_linevec_faster_4_stages.prototxt
      - pose_iter_160000.caffemodel
      - Output.jpg - White background image
      - Model created and its name will be stored in folder with a name of input video name.
      - Train data file created will be stored here
2) You have to change the Output location to as per your system.
3) This code can be run on windows operating system if you want to make use of it for other operating system in that case you may required to change hardcoded paths mention in Code.
4) Below packages should be present on your system to run below code:
import os 
import cv2
import glob
import time
import shutil
from matplotlib import pyplot as plt
import pandas as pd
import requests
import json
import ast
import time
import statistics


Once all the requirements are full filled you can try to run this code on your machine.
