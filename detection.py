import jetson.inference
import jetson.utils

model = "ssd-mobilenet-v2"; threshold = 0.25
webcam = "/dev/video1"

net = jetson.inference.detectNet(network=model, threshold=threshold)
camera = jetson.utils.videoSource(webcam)
display = jetson.utils.videoOutput()

while True:
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
