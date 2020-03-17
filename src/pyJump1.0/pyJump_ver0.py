"""PY跳棋"""

import pygame, sys, math
from pygame.locals import *

WINWIDTH, WINHEIGHT = 800, 600
CENTERX, CENTERY = WINWIDTH//2, WINHEIGHT//2

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

BGCOLOR = BLACK # syntactic sugars
BLANK_COLOR = WHITE

class Point:
    a = 35 # 等边三角形的边长
    e1 = (a, 0)
    e2 = (-a/2, a*math.sqrt(3)/2)
    radius = 10 # 点的半径
    O = (CENTERX, CENTERY) # 自建坐标系的原点，像素坐标系的中心
    def __init__(self, x=0, y=0):
        self.x = x # x, y并非直角坐标
        self.y = y # 这个点的坐标 = x*e1 + y*e2, 初始值是原点

        x = Point.O[0] + self.x*Point.e1[0] + self.y*Point.e2[0]  # 注意: self.x和x的意义不同
        y = Point.O[1] -(self.x*Point.e1[1] + self.y*Point.e2[1]) # x是像素直角坐标系的x坐标, self.x是自建坐标系的x坐标
                                                                  # 此外，在像素坐标系中，y越大，越往下，而不是往上
        self.pos = (x, y) # 这个点在像素坐标系上的坐标
        left, top = self.pos[0]-Point.radius, self.pos[1]-Point.radius
        width, height = Point.radius, Point.radius
        self.rect = pygame.Rect(left, top, width, height) # 圆圈所对应的矩形

        self.color = BLANK_COLOR # 这个点现在是什么颜色，初始为BLANK_COLOR，会在调用draw的时候改变

    def draw(self, surf, color, point=None):
        """绘制自己。"""
        if not point:
            # 如果没有传入point参数，使用self.pos
            point = (int(self.pos[0]), int(self.pos[1])) # 圆心，坐标必须是整型
            self.rect = pygame.draw.circle(surf, color, point, Point.radius, 0) # self.rect在调用这个函数的时候会更新
            self.color = color
        else:
            # 传入了point参数，主要是适用于绘制不在自建坐标系上的点的情况
            # 不更新self.rect和self.color
            pygame.draw.circle(surf, color, point, Point.radius, 0)

def create_empty_pDict():
    """最基本的空棋盘，包括棋盘上所有的点的位置信息。"""
    pSet  = set() # 用不同的数据结构来存储
    pDict = {}    # 棋盘上所有的点
    for x in range(-4, 5): # 一行一行地添加
        pSet.add((x, 0))
    for x in range(-4, 6):
        pSet.add((x, 1))
    for x in range(-4, 7):
        pSet.add((x, 2))
    for x in range(-4, 8):
        pSet.add((x, 3))
    for x in range(-4, 9):
        pSet.add((x, 4))
    for x in range(1, 5):
        pSet.add((x, 5))
    for x in range(2, 5):
        pSet.add((x, 6))
    for x in range(3, 5):
        pSet.add((x, 7))
    for x in range(4, 5):
        pSet.add((x, 8))

    for x,y in list(pSet): # 强行list一下，不然在迭代过程中pSet会改变，会报错
        pSet.add((-x, -y))

    for x,y in pSet:
        pDict[(x,y)] = Point(x,y) # pDict是一个映射，从自建坐标(x,y)到对应的Point实例
    return pDict
    
def draw_board(pDict):
    """绘制棋盘。使用pDict中记录的颜色。"""
    global screen
    screen.fill(BGCOLOR)
    for p in pDict.values():
        p.draw(screen, p.color)

def create_2player_pDict():
    pDict = create_empty_pDict()
    colorA = BLUE
    colorB = RED

    AList = [
                 (4,8),
              (3,7),(4,7),
           (2,6),(3,6),(4,6),
        (1,5),(2,5),(3,5),(4,5),
        ] # 玩家A的点的初始位置
    BList = []
    for x,y in AList:
        BList.append((-x,-y)) # 玩家B的点的初始位置

    for x,y in AList:
        pDict[(x,y)].color = colorA
    for x,y in BList:
        pDict[(x,y)].color = colorB
    return pDict

def isValidMove(p0, p, pDict):
    """
    布尔函数，判断落子的位置是否合法，是整个程序的核心逻辑。
    p0是移动棋子之前的点，p是要判断的移动棋子后的点。
    """
    def create_initial_vpSet():
        # 第一步：生成一个包含自己和邻居的合法点集
        # 'vp' stands for 'valid points'
        vpSet = set() # 所有有效（合法）点--以向量表示，因为纯粹是一个涉及坐标的问题，与颜色等其他信息无关--的集合
        x, y = p0.x, p0.y
        vpSet.add((x,y)) # 先把自己放进去

        # 再把自己的邻居放进去
        neighbors = [
                (0,1),    (1,1),

            (-1,0),           (1,0),

                (-1,-1),  (0,-1),
            ]
        for (offset, vector) in enumerate(neighbors):
            neighbors[offset] = (vector[0]+x, vector[1]+y)
        vpSet = set.union(vpSet, {(x,y) for (x,y) in neighbors if (x,y) in pDict.keys()})
        return vpSet
    def create_clpSets():
        # 第二步：生成一个包含所有与自己共线的点的点集
        # 'clp' stands for co-linear points

        # 第一种情况: x相等, y不相等（若y相等，则就是自己，也包含在内
        # 第二种情况：y相等, x不相等（若x相等，同上）
        # 第三种情况：p0.x - x = p0.y - y
        # 把三种情况分开，方便第三步的处理
        clpSet1 = ((x,y) for (x,y) in pDict.keys() if x == p0.x)
        clpSet2 = {(x,y) for (x,y) in pDict.keys() if y == p0.y}
        clpSet3 = {(x,y) for (x,y) in pDict.keys() if (p0.x-x) == (p0.y-y)}
        return [clpSet1, clpSet2, clpSet3]
    def delete_blank_and_far_points(clpSets):
        # 第三步：删去clpSet中没有棋子的点，然后每个方向仅保留最近的点
        # 没有棋子，意味着颜色不为白色
        # 'vclp' stands for valid co-linear points
        # 三种情况，亦即三个方向，分别处理，最后合并，最多只保留六个点（每个方向两个）

        vclpSet = set()

        # vclpSet1 对应着x坐标相等的点集
        # 比较y的差值即可，正负分别比较
        vclpSet1 = {(x,y) for (x,y) in clpSet1 if pDict[(x,y)].color != WHITE}
        dy_max = 12 # 12是棋盘上两个点的最远距离
        for dy in range(1, dy_max+1):
            x,y = p0.x, p0.y+dy
            if pDict[(x,y)].color != WHITE:
                vclpSet.add((x,y))
                break # 找到第一个满足条件的，就是最近的，直接退出循环
                      # 如果没有满足条件的，则走完整个for循环
        for dy in range(-1, -(dy_max+1), -1): # 反方向来一遍
            x,y = p0.x, p0.y+dy
            if pDict[(x,y)].color != WHITE:
                vclpSet.add((x,y))
                break

        # vclpSet2 对应着y坐标相等的点集
        vclpSet2 = {(x,y) for (x,y) in clpSet2 if pDict[(x,y)].color != WHITE}
        '''
        pdy = 12   # pdy, ndy stands for positive, negative
        ndy = -pdy # 12是棋盘上两个点的最远距离
        for p in vclpSet1:
            dy = p[1] - p0.y
            if dy == 0:
                # 记得先把自己删掉，因为第二步中没有删掉
                pass
            elif 0 < dy < pdy:
                pdy = dy
            elif ndy < dy < 0:
                ndy = dy
        '''

    def create_tpSet(vclpSet):
        # 第四步：生成所有可能的目标点tp
        # 注意：起点p0和目标点tp关于vclp对称，把这个关系转换成向量即可
        # 'tp' stands for target points
        tpSet = set()
        for vclp in vclpSet:
            vector = (vclp[0]-p0.x, vclp[1]-p0.y)
            tp = (vclp[0]+vector[0], vclp[1]+vector[1])
            tpSet.add(tp)
        return tpSet
    def delete_invalid_points(tpSet):
        # 第五步：删去tpSet中非法的点
        # 非法的点包括：
        #     * 有棋子的点，所以要留住颜色为白色的点
        #     * 棋盘外的点
        # 注意：要先检测(x,y) in pDict.keys()，否则会引发KeyError
        return {(x,y) for (x,y) in tpSet if (x,y) in pDict.keys() and pDict[(x,y)].color == WHITE}

    vpSet  = create_initial_vpSet()
    clpSets = create_clpSets()
    vclpSet = delete_blank_points(clpSets)
    tpSet = create_tpSet(vclpSet)
    tpSet = delete_invalid_points(tpSet)

    for (x,y) in tpSet:
        Point(x,y).draw(screen, YELLOW)
    pygame.display.update()
    pygame.time.wait(1000)

    return (p.x, p.y) in vpSet

def terminate():
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    screen = pygame.display.set_mode((WINWIDTH, WINHEIGHT))

    FPSCLOCK = pygame.time.Clock()
    FPS = 30

    pDict = create_2player_pDict()
    draw_board(pDict)
    currentChequer = None # 现在正在拿着的棋子，没有时为None，有时为Point实例
    currentColor = BLANK_COLOR # 记录currentChequer的颜色, 初始为BLANK_COLOR
    while True:
        for event in pygame.event.get(QUIT): terminate()
        for event in pygame.event.get(MOUSEBUTTONUP):
            if currentChequer and event.button == 3:
                # 这句话是后加的，逻辑不是按顺序的
                # 之所以放在前面是为了更加灵敏地反应

                # 拿着子儿呢，同时点击鼠标右键：放下棋子
                print('drop')
                assert currentColor != BLANK_COLOR
                currentChequer.color = currentColor # 恢复原来的颜色
                currentColor = BLANK_COLOR
                currentChequer = None # 更新状态
                draw_board(pDict)

            for x,y in pDict.keys():
                # x,y 是自建坐标系的坐标
                p = pDict[(x,y)] # p是Point类的实例
                if p.rect.collidepoint(event.pos):
                    # 碰撞检测: 鼠标按键位置是棋盘上的某个点
                    if not currentChequer and p.color != BLANK_COLOR:
                        # 没拿着子儿，同时鼠标的位置所在有子儿（颜色非白色）
                        if event.button == 1: # 鼠标左键：拿起一个棋子
                            print('pick up')
                            currentChequer = p # 更新状态
                    elif currentChequer and event.button == 1 and \
                         isValidMove(currentChequer, p, pDict): # p是移动的目标点
                        # 拿着子儿呢，同时点击鼠标左键，落子
                        # 注意“放下棋子”和“落子”的区别，前者是放回原处，后者是放到新的地方
                        print('落子')
                        # 注意: 这里的(x,y)是碰撞检测的结果，也就是新的落子点--目标点
                        assert currentColor != BLANK_COLOR
                        pDict[(x,y)].color = currentColor # 改变目标点的颜色
                        currentColor = BLANK_COLOR
                        currentChequer = None # 更新状态
                        draw_board(pDict)

        if currentChequer:
            # 拿着子儿呢
            p = currentChequer # syntactic sugar

            # 注意: 棋子的颜色不可能是空白色(BLANK_COLOR)，所以如果是空白色，肯定是因为程序后面的处理
            if p.color != BLANK_COLOR:
                currentColor = p.color
                p.color = BLANK_COLOR # 注意：拿起来之后，原来的位置就变成空白色
                # 这样做还有一个好处，就是如果把子落到新的位置上，就不需要对原来的位置再作处理
            else:
                # 如果p.color已经是空白色，此时currentColor一定不是空白色，就不要再改变它的值了
                assert p.color == BLANK_COLOR and currentColor != BLANK_COLOR
            for event in pygame.event.get(MOUSEMOTION):
                # 只有在拿着棋子的时候，才需要处理MOUSEMOTION类事件
                draw_board(pDict)
                Point.draw(p, screen, currentColor, point=event.pos)
        for event in pygame.event.get():
            if event.type == KEYUP and event.key == K_ESCAPE: terminate()
        pygame.display.update()
        FPSCLOCK.tick(FPS)
