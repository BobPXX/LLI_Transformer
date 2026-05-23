path="C:/Users/xiuxi/Desktop/"
import cv2
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import os
import numpy as np
import sys
import time
import screeninfo

#crop [y0:y1, x0:x1],[750:2350, 1450:3050]
y0=350
y1=1950
x0=900
x1=2500

screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height
window_name = ''
window_size=500
screen_x=700
screen_y=490

#open device-----------------------------------------------------------
ids_peak.Library.Initialize()
device_manager = ids_peak.DeviceManager.Instance()
device_manager.Update()
camera=device_manager.Devices()[0]
device = camera.OpenDevice(ids_peak.DeviceAccessType_Control)

datastreams = device.DataStreams()
datastream = datastreams[0].OpenDataStream()

nodemap_remote_device = device.RemoteDevice().NodeMaps()[0]
nodemap_remote_device.FindNode("UserSetSelector").SetCurrentEntry("Default")
nodemap_remote_device.FindNode("UserSetLoad").Execute()
nodemap_remote_device.FindNode("UserSetLoad").WaitUntilDone()
nodemap_remote_device.FindNode("ExposureTime").SetValue(100000)#default 30000

payload_size = nodemap_remote_device.FindNode("PayloadSize").Value()
buffer_count_max = datastream.NumBuffersAnnouncedMinRequired()
for i in range(buffer_count_max):
    buffer =datastream.AllocAndAnnounceBuffer(payload_size)
    datastream.QueueBuffer(buffer)
#--------------------------------------------------------------------
#3 buffers are already in InputBufferPool before start
#start aquisition
datastream.StartAcquisition()
nodemap_remote_device.FindNode("AcquisitionStart").Execute()
nodemap_remote_device.FindNode("AcquisitionStart").WaitUntilDone()
#3 buffers are all filled

buffer = datastream.WaitForFinishedBuffer(5000) #send one buffer from InputBufferPool to OutputBufferPool
buffer = datastream.WaitForFinishedBuffer(5000) #send another buffer from InputBufferPool to OutputBufferPool
buffer = datastream.WaitForFinishedBuffer(5000) #send last buffer from InputBufferPool to OutputBufferPool, no buffer in InputBufferPool now
#-------------------------------------------------------------------------------------------------
def perform(dataset_add, pattern_add,RGB):
    # show
    img = cv2.imread(dataset_add)
    cv2.namedWindow(window_name, 0);
    cv2.resizeWindow(window_name, window_size, window_size);
    cv2.moveWindow(window_name, screen.x + screen_x, screen.y + screen_y)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

    # wait
    time.sleep(0.3)

    # shot
    global buffer
    datastream.QueueBuffer(buffer)  # add a new buffer in InputBufferPool, it was filled by the device immediate since StartAquisition() was opened
    buffer = datastream.WaitForFinishedBuffer(5000)  # send this buffer from InputBufferPool to OutputBufferPool
    # Transform buffer (in OutputBufferPool) to image
    ipl_image = ids_peak_ipl.Image_CreateFromSizeAndBuffer(
        buffer.PixelFormat(),
        buffer.BasePtr(),
        buffer.Size(),
        buffer.Width(),
        buffer.Height()
    )
    if RGB==True:
        converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB12) 
    else:
        converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_Mono12)

    frame = converted_ipl_image.get_numpy_3D_16()
    cv2.destroyAllWindows()

    # save
    crop = frame[y0:y1, x0:x1]  # [y0:y1, x0:x1]
    #resize = cv2.resize(crop, (resize_size, resize_size))
    np.save(pattern_add, crop)
#-------------------------------------------------------------------------------------------------
print('fruits...start...')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

dataset_path=path+'dataset/fruits_modified/'
classes=os.listdir(dataset_path)
classes.sort()

pattern_path=path+'pattern/fruits_modified/'
if not os.path.exists(pattern_path):
    os.mkdir(pattern_path)
for cl in range(len(classes)):
    if not os.path.exists(pattern_path+classes[cl]):
        os.mkdir(pattern_path+classes[cl])

all_files= [ [ None for y in range(1665) ]
             for x in range(len(classes)) ]
for cl in range(len(classes)):
    files=os.listdir(dataset_path+classes[cl])
    files.sort()
    for i in range(len(os.listdir(dataset_path+classes[cl]))):
        all_files[cl][i]=files[i]

for i in range(1664):
    for cl in range(len(classes)):
        if all_files[cl][i]!=None:
            dataset_add=dataset_path+classes[cl]+'/'+all_files[cl][i]
            pattern_add=pattern_path+classes[cl]+'/'+all_files[cl][i][:-5]+'.npy'
            perform(dataset_add,pattern_add,RGB=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#stop aquisition
remote_nodemap = device.RemoteDevice().NodeMaps()[0]
remote_nodemap.FindNode("AcquisitionStop").Execute()
datastream.KillWait()
datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

for buffer in datastream.AnnouncedBuffers():
    datastream.RevokeBuffer(buffer)