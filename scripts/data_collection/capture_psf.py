path="C:/Users/xiuxi/Desktop/pattern/"
import cv2
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import os
import numpy as np
import sys

#resize_size=224
#crop [y0:y1, x0:x1],[750:2350, 1450:3050]
y0=350
y1=1950
x0=900
x1=2500


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
nodemap_remote_device.FindNode("ExposureTime").SetValue(30000)#psf 100000 #img 30000

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



# shot
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
converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB12)
frame = converted_ipl_image.get_numpy_3D_16()

# save
cv2.imwrite(path + 'psf_ori.bmp', cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX))
np.save(path + 'psf_ori.npy', frame)

crop = frame[y0:y1, x0:x1]  # [y0:y1, x0:x1]
#resize = cv2.resize(crop, (resize_size, resize_size))
cv2.imwrite(path + 'psf.bmp', cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX))
np.save(path + 'psf.npy', crop)



#stop aquisition
remote_nodemap = device.RemoteDevice().NodeMaps()[0]
remote_nodemap.FindNode("AcquisitionStop").Execute()
datastream.KillWait()
datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

for buffer in datastream.AnnouncedBuffers():
    datastream.RevokeBuffer(buffer)