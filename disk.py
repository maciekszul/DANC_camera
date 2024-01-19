import psutil
import time

while True:
    print("Percentage used /video_data: ", psutil.disk_usage("/video_data").percent)
    print("Percentage used /ssd_data: ", psutil.disk_usage("/ssd_data").percent)
    print("\n")
    time.sleep(0.5)
