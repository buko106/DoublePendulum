import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse

parser = argparse.ArgumentParser(description="Double pendulum Simulator.")
parser.add_argument("-iter", default=3000, type=int, help="The number of interation")
parser.add_argument("-step", default=0.01, type=float, help="Runge-Kutta method's step-time")
parser.add_argument("-sampling", default=5, type=int, help="Draw a frame by SAMPLING steps")
parser.add_argument("-interval", default=50, type=float, help="Interval time(ms) between a frame and next one")
parser.add_argument("-o", default=None, type=str, help="Outut file(can be .gif or .mp4)")
parser.add_argument("-g", default=9.8, type=float, help="Gravitational acceleration")
parser.add_argument("-l1", default=np.random.uniform(1,4), type=float)
parser.add_argument("-l2", default=np.random.uniform(1,4), type=float)
parser.add_argument("-m1", default=np.random.uniform(1,10), type=float)
parser.add_argument("-m2", default=np.random.uniform(1,10), type=float)
parser.add_argument("-t1", default=np.random.uniform(-np.pi,np.pi), type=float)
parser.add_argument("-dt1", default=np.random.uniform(-np.pi,np.pi), type=float)
parser.add_argument("-t2", default=np.random.uniform(-np.pi,np.pi), type=float)
parser.add_argument("-dt2", default=np.random.uniform(-np.pi,np.pi), type=float)
parser.add_argument('--without-line', action='store_true', default=False, dest="without_line")

args = parser.parse_args()

g  = args.g
l1 = args.l1
l2 = args.l2
m1 = args.m1
m2 = args.m2
n  = args.iter
h  = args.step
a1 = (m1+m2) * (l1**2)
a2 = m2 * (l2**2)

def f(x):
    dt = x[0]-x[2]
    b = m2 * l1 * l2 * np.cos(dt)
    d1 = -m2 * l1 * l2 * x[3] * x[3] * np.sin(dt) - (m1+m2) * g * l1 * np.sin(x[0])
    d2 =  m2 * l1 * l2 * x[1] * x[1] * np.sin(dt) - m2 * g * l2 * np.sin(x[2])
    return np.array([x[1],
                     (a2*d1-b*d2)/(a1*a2-b*b),
                     x[3],
                     (a1*d2-b*d1)/(a1*a2-b*b)])

result = []
times  = []
x = np.array([args.t1,args.dt1,args.t2,args.dt2])
for i in xrange(n):
    t = h * i
    times  += [t]
    result += [x]
    # RK4
    p  = x
    k1 = f(p)
    p  = x + k1 * (h * 0.5)
    k2 = f(p)
    p  = x + k2 * (h * 0.5)
    k3 = f(p)
    p  = x + k3 * h
    k4 = f(p)
    
    x = x + ( k1 + 2.0*k2 + 2.0*k3 + k4 )*(h/6.0)


result = np.array(result)
times = np.array(times)

xy1 = np.array([np.sin(result[:,0]),-np.cos(result[:,0])]) * l1
xy = xy1 + np.array([np.sin(result[:,2]),-np.cos(result[:,2])]) * l2


images = []
l = l1+l2
fig = plt.figure()
plt.axes(xlim=(-l,+l),ylim=(-l,+l))
plt.gca().set_aspect('equal', adjustable='box')

for i in xrange(0,n,args.sampling):
    im = []
    if not args.without_line:
        im += plt.plot(xy[0,:i],xy[1,:i],"-",lw=0.5,color="blue")
    im += plt.plot([0,xy1[0,i]],[0,xy1[1,i]],"-",lw=1,color="magenta")
    im += plt.plot([xy1[0,i],xy[0,i]],[xy1[1,i],xy[1,i]],"-",lw=1,color="red")
    im += plt.plot(xy1[0,i],xy1[1,i],".",ms=m1*5,color="magenta")
    im += plt.plot(xy[0,i],xy[1,i],".",ms=m2*5,color="red")
    images.append(im)

    

ani = animation.ArtistAnimation(fig,images,repeat=True,interval=args.interval)

if args.o:
    _, ext = os.path.splitext(args.o)
    if ext == ".gif":
        ani.save(args.o, writer="imagemagick")
    elif ext == ".mp4":
        ani.save(args.o, writer="ffmpeg")
else :
    plt.show()
