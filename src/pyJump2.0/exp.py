from pyJump_ver9 import *
import pygame

class Point:
    radius = 20
    O = (CENTERX, CENTERY)
    vmax = 2
    def __init__(self, sx=0, sy=0):
        # 位移, 相对于原点
        self.sx = sx
        self.sy = sy
        # velocity
        self.vx = 0
        self.vy = 0
        # acceleration
        self.ax = 0.2
        self.ay = 0.2
    def redraw(self, surf, color):
        pos = int(Point.O[0]+self.sx), int(Point.O[1]-self.sy) # 取整, sy取负值以保证"y越大，位于屏幕越高位置"
        pygame.draw.circle(surf, color, pos, Point.radius)

if __name__ == '__main__':
    screen = pygame.display.set_mode((WINWIDTH, WINHEIGHT))

    FPSCLOCK = pygame.time.Clock()
    FPS = 30

    A = Point(-10, 0)
    B = Point(10, 0)

    runningUp = False
    runningDown = False
    runningLeft = False
    runningRight = False

    while True:
        screen.fill(BLACK)

        A.redraw(screen, RED)
        #B.redraw(screen, BLUE)

        for event in pygame.event.get(KEYDOWN):
            if event.key == K_UP: runningUp = True
            elif event.key == K_DOWN: runningDown = True
            elif event.key == K_LEFT: runningLeft = True
            elif event.key == K_RIGHT: runningRight = True
        for event in pygame.event.get(KEYUP):
            if event.key == K_UP: runningUp = False
            elif event.key == K_DOWN: runningDown = False
            elif event.key == K_LEFT: runningLeft = False
            elif event.key == K_RIGHT: runningRight = False
            else: pygame.event.post(event)

        if runningUp: # 根据e1的定义方式, y越大，越靠上，所以speed为正时向上运动
            if A.vy == Point.vmax: pass
            else:
                A.vy += A.ay # 有一个加速的过程
                if A.vy > Point.vmax: A.vy = Point.vmax # 浮点数的计算有点小问题
        else:
            if A.vy == 0: pass
            else:
                A.vy -= A.ay
                if A.vy < 0: A.vy = 0

        '''
        if runningLeft: Ax -= Aspeed
        if runningRight: Ax += Aspeed
        '''

        A.sx += A.vx
        A.sy += A.vy
        '''
        # B的移动
        if Bx-Ax > 0: Bx -= Bspeed
        else: Bx += Bspeed
        if By-Ay > 0: By -= Bspeed
        else: By += Bspeed
        '''

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE): terminate()
        pygame.display.update()
        FPSCLOCK.tick(FPS)
