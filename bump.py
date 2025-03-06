import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from dataclasses import dataclass

def packet(r):
    return r/(1+r**4)

def wavefront(r,t):
    return packet(r-t)-packet(r+t)

def wavefront_2d(r,t):
    return wavefront(r,t)/np.sqrt(t+.001)

@dataclass
class Source:
    x: float
    y: float
    t: float
    height: float

sources = [Source(0,0,-1,0)]

def wave(x, y, t):
    z = 0
    for source in sources:
        if t < source.t:
            continue
        z += source.height * wavefront_2d(np.sqrt((source.x-x)**2+(source.y-y)**2), t-source.t)
    return z

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    period: float
    m: float
    wave_height: float
    next_bounce: float = 0
    last_update_t: float = 0

    def _step(self, dt):
        self.x += dt * self.vx
        self.y += dt * self.vy

    def _bounce(self):
        global sources
        t = self.next_bounce
        dx = 0.01
        dy = 0.01
        self.vx -= 1/self.m*(wave(self.x+dx,self.y,t)-wave(self.x-dx,self.y,t))/(2*dx)
        self.vy -= 1/self.m*(wave(self.x,self.y+dy,t)-wave(self.x,self.y-dy,t))/(2*dy)
        sources.append(Source(self.x, self.y, t, self.wave_height))
        self.next_bounce = t + self.period

    def update(self, t):
        if t > self.next_bounce:
            self._step(self.next_bounce - self.last_update_t)
            time_after = t - self.next_bounce
            self._bounce()
            self._step(time_after)
        else:
            self._step(t - self.last_update_t)
        self.last_update_t = t

particles = [
    Particle(x=0, y=0, vx=0, vy=0, period=1, m=100, wave_height=1),
    Particle(x=2, y=0, vx=0, vy=0.3, period=2, m=2, wave_height=0),
    Particle(x=4, y=0, vx=0, vy=0.3, period=2, m=2, wave_height=0),
]

def center(x, y):
    for particle in particles:
        particle.x -= x
        particle.y -= y
    for source in sources:
        source.x -= x
        source.y -= y

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(10,10))

def animate(frame):
    t = frame/30
    # center(particles[0].x, particles[0].y)
    for particle in particles:
        particle.update(t)
    z = wave(X, Y, t)
    ax.cla()
    ax.pcolormesh(X, Y, z, vmin=-.2, vmax=.2, cmap='plasma')
    ax.scatter([p.x for p in particles], [p.y for p in particles], c='white', marker='x')
    ax.scatter([s.x for s in sources], [s.y for s in sources], c='blue', marker='o')
    ax.grid(True)

ani = animation.FuncAnimation(fig, animate, frames=10000, interval=0, repeat=True)

plt.show()
