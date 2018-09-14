import sys, pygame
import argparse
import math
from pythonosc import dispatcher
from pythonosc import osc_server
import numpy as np
import threading
from queue import Queue

notmovingwhere = 0
leftwhere = 1
rightwhere = 2
upwhere = 3
downwhere = 4
leftblinkwhere = 5
rightblinkwhere = 6


vertical_buffer=np.zeros(200).tolist()
horizontal_buffer=np.zeros(200).tolist()

movement = 200

def movement_func(direction):
    if direction == notmovingwhere:
        vertical_buffer.append(0)
        horizontal_buffer.append(0)
    elif direction == leftwhere:
        vertical_buffer.append(0)
        horizontal_buffer.append((-1)*movement)
    elif direction == rightwhere:
        vertical_buffer.append(0)
        horizontal_buffer.append(movement)
    elif direction == upwhere:
        vertical_buffer.append((-1)*movement)
        horizontal_buffer.append(0)
    elif direction == downwhere:
        vertical_buffer.append(movement)
        horizontal_buffer.append(0)
    elif direction == leftblinkwhere:
        vertical_buffer.append(-1 * movement)
        horizontal_buffer.append(-1 * movement)
    elif direction == rightblinkwhere:
        vertical_buffer.append(movement)
        horizontal_buffer.append(movement)
    del vertical_buffer[0]
    del horizontal_buffer[0]

    return np.mean(vertical_buffer), np.mean(horizontal_buffer)

'''
weight_1 = np.array([[0.0773,  0.0835,  0.0001, -0.1405],
                    [0.0312,  0.1036, -0.0197, -0.0937],
                    [-0.0212, -0.0715,  0.1677, -0.0675],
                    [-0.0449, -0.0648,  0.0962,  0.0621],
                    [-0.0405,  0.1250, -0.0250, -0.0534],
                    [-0.0748,  0.0703,  0.0131,  0.0040],
                    [-0.0035, -0.0038, -0.0035, -0.0035],
                    [-0.0211, -0.0854,  0.1134,  0.0354],
                    [0.0466,  0.0342, -0.0718,  0.0696],
                    [0.1267,  0.0091, -0.0824,  0.0186]])

weight_2 = np.array([[0.0057, -0.0151,  0.0123, -0.0012, -0.1101, -0.0406,  0.0001,  0.0134, -0.0040,  0.0252],
                    [0.1443,  0.1427, -0.1132, -0.3405,  0.0740, -0.0327, -0.0031, -0.1260, 0.0392,  0.0571],
                    [-0.3360, -0.3193,  0.1062,  0.1956,  0.0670,  0.0954, -0.0020,  0.0828, -0.0540, -0.1347],
                    [0.0282,  0.0534,  0.0021, -0.0044,  0.0380,  0.0849,  0.0019, -0.0127, -0.0495, -0.0514],
                    [-0.2092, -0.2309, -0.3739, -0.0087, -0.3470, -0.2005, -0.0025, -0.0064,  0.0814, 0.0734],
                    [0.0477,  0.1064, -0.1305, -0.1240,  0.1173,  0.0992,  0.0035, -0.0759,  0.0105, -0.0329],
                    [0.2012,  0.0230,  0.2087, 0.0513, -0.2065, -0.1755, -0.0023,  0.0608, -0.1589, -0.0333]])
bias_1 = np.array([[6.9574e-01, 6.9040e-01, 7.7783e-01, 8.4647e-01, 7.6082e-01, 7.5973e-01, 3.7605e-07, 5.5906e-01, 7.2283e-01, 6.4912e-01]])
bias_2 = np.array([[0.9897, 1.0100, 1.0660, 1.0746, 0.9166, 0.8587, 0.7536]])
'''

weight_1 = np.array([[0.1108, 0.0268, -0.0384, -0.2651],
                     [0.1549, -0.2104, -0.1201, -0.1617],
                     [-0.0780, -0.3444, 0.1215, 0.0288],
                     [-0.0103, 0.0223, -0.0238, 0.0034],
                     [-0.0102, 0.2662, -0.1693, -0.2048],
                     [-0.1939, -0.1065, 0.1299, 0.1378],
                     [0.0201, 0.1657, 0.0170, 0.1782],
                     [0.0392, -0.1136, 0.0075, 0.2290],
                     [0.1016, -0.0025, -0.0493, -0.0923],
                     [0.3022, 0.1266, -0.3651, 0.1946]])

weight_2 = np.array([[-0.2391, 0.1733, -0.0835, 1.0489, -0.0996, 0.1485, 0.1406, 0.0206, 0.5965, -0.1803],
                     [0.2121, -0.8067, -0.5810, -0.7080, 0.0680, -0.2505, -0.1640, -0.0914, 0.3027, 0.1503],
                     [0.0987, -0.1438, 0.0867, -0.5270, -0.4740, 0.1723, -0.6368, 0.1099, -0.3911, 0.2236],
                     [0.0956, 0.2113, 0.1692, 0.1299, 0.0848, -0.1624, -0.0137, -0.7066, -0.6118, -0.4748],
                     [-0.3041, 0.0183, -0.1871, -0.8306, 0.0532, -0.1163, 0.1898, 0.1210, -0.3746, 0.0501]])

bias_1 = np.array([[0.2692, 1.6755, 0.0673, 2.3516, 0.2724, 0.8290, 1.6424, 0.2392, 2.3817, 0.1022]])

bias_2 = np.array([[1.8916, 0.2406, 0.2860, 0.6593, -0.0824]])

reference_value = np.array([[816.921, 837.52905, 847.6728, 821.1709]])    # zhangge

def eeg_handler(unused_addr, args, data1,data2,data3,data4):
    global weight_1
    global weight_2
    global bias_1
    global bias_2
    global reference_value
    global position
    input_data = np.array([data1,data2,data3,data4]).reshape(1,4)
    pro = np.dot(np.maximum(0,np.dot(input_data - reference_value, weight_1.T) + bias_1),weight_2.T) + bias_2
    index = np.argwhere(pro == np.amax(pro))[0][1]
    position = index
    #print(index)

pygame.init()
size = width, height = 800, 600
speed = [2, 2]
black = 0, 0, 0

screen = pygame.display.set_mode(size)

ball = pygame.image.load("ball.bmp")
ballrect = ball.get_rect()
position = 0

def update_data():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="127.0.0.1",
                        help="The ip to listen on")
    parser.add_argument("--port",
                        type=int,
                        default=21000,
                        help="The port to listen on")
    args = parser.parse_args()
                        
    dispatcher1 = dispatcher.Dispatcher()
    dispatcher1.map("/debug", print)
    dispatcher1.map("/muse/eeg", eeg_handler,"eeg")
                        
    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher1)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()


t=threading.Thread(target=update_data)
t.daemon=True
t.start()

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    
    horizontal_move = movement_func(position)[1]
    vertical_move = movement_func(position)[0]

    ballrect.left = width/2-125 + horizontal_move
    ballrect.top = height/2-125 + vertical_move

    screen.fill(black)
    screen.blit(ball, ballrect)
    pygame.display.flip()
    #print("H", horizontal_buffer)
    #print("V", vertical_buffer)

t.join()
