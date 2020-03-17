"""
PY跳棋

ver2
实现了游戏的核心逻辑isValidMove

ver3
调整了前述函数的结构，
并实现了高光与拿起棋子同步。

ver4
引入了玩家的逻辑，
一人一步，交替进行游戏。

ver5
优化，在create_vpSet的递归阶段，不能把自己算作支点，
traceDict初步建立。

ver6
第一版完成。
"""

import pygame, sys, math
from pygame.locals import *

WINWIDTH, WINHEIGHT = 800, 600
CENTERX, CENTERY = WINWIDTH//2, WINHEIGHT//2

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

colorA = BLUE # 玩家的颜色
colorB = RED

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

        self.color = WHITE # 这个点现在是什么颜色，初始为WHITE，会在调用draw的时候改变

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
    def get_coord(self):
        """返回自己的自建坐标。"""
        return (self.x, self.y)

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
    r,g,b = playerList[currentPlayer]['color'] # 当前玩家的颜色
    color = r//3, g//3, b//3 # 弱化颜色，否则看不见当前玩家的棋子
    screen.fill(color) # 以当前玩家的颜色为底色，以提醒每位玩家
    for p in pDict.values():
        p.draw(screen, p.color)

def create_2player_pDict():
    pDict = create_empty_pDict()

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

def create_2player_playerList(pDict):
    """
    玩家以字典存储，有两个键:
        * 颜色
        * 目标，也就是为了获得胜利而需要到达的所有点的位置
    注意: pDict必须是刚刚建立、未经变动的原始值，也就是函数create_2player_pDict的返回值
    """
    playerA = {'color': colorA}
    playerA['goals'] = []
    playerB = {'color': colorB}
    playerB['goals'] = []
    # 由于两人游戏的特殊性，双方的目标都是对方的老家
    # 所以利用初始的pDict即可创建玩家的目标键值
    for (x,y) in pDict.keys():
        if pDict[(x,y)].color == colorA:
            playerB['goals'].append((x,y))
        if pDict[(x,y)].color == colorB:
            playerA['goals'].append((x,y))
    return [playerA, playerB]

def create_vpSet(p0, pDict, traceDict):
    """
    从p0出发，有多少种可能的走法？
    这个函数将告诉你答案。
    traceDict是在调用此函数之前预先创建的字典，currentTraceDict
    这个函数并不关心它在传入时的值，
    只是利用python对字典的引用，在这个函数返回时，相应地更新traceDict的值（不以返回值的形式）
    """
    def create_initial_vpSet():
        # 生成一个包含自己邻居的点集
        # 'vp' stands for 'valid points'
        # vpSet是所有有效（合法）点--以向量表示，因为纯粹是一个涉及坐标的问题，与颜色等其他信息无关--的集合
        # 这一步只是生成了一个初始的vpSet，非法的点将在第四步除去
        x, y = p0.x, p0.y

        # 把自己的邻居放进去
        neighbors = [
                (0,1),    (1,1),

            (-1,0),           (1,0),

                (-1,-1),  (0,-1),
            ]
        for (offset, vector) in enumerate(neighbors):
            neighbors[offset] = (vector[0]+x, vector[1]+y)
        vpSet = {(x,y) for (x,y) in neighbors}

        # 更新父子关系
        nonlocal pcDict
        pcDict[(p0.x, p0.y)] = set(neighbors)       # 更新父子关系
                                                    # 注意: 尽管有的邻居是非法点，但这并不影响轨迹的绘制
        for (x,y) in neighbors:
            pcDict[(x,y)] = set() # 邻居不可能是别人的父，但没有子的父也是父

        return vpSet
    def create_fSet(p):
        # 生成一个包含所有支点的点集。
        # 支点具有下列性质：
        #     * 与p共线;
        #     * 点上有棋子;
        #     * 与p之间没有别的棋子，也就是，在该方向距离p最近（共六个方向）;
        #     * 与p的镜像之间也没有棋子

        # 共线分为三种情况，分别讨论，每条线上两个方向，共六个方向
        # 没有棋子，意味着颜色不为白色
        # 'f' stands for fulcrum
        # 最多只有六个点（每个方向两个）
        # 注意，当用于递归时，参数p不是起点p0

        fSet = set()

        # 第一种情况：x坐标相等
        # 比较y的差值即可，正负分别比较
        dy_max = 12 # 12是棋盘上两个点的最远距离
        for dy in range(1, dy_max+1):
            x,y = p.x, p.y+dy
            if (x,y) not in pDict.keys(): break # 超出棋盘，直接break
            if pDict[(x,y)].color != WHITE:
                fSet.add((x,y))
                break # 找到第一个满足条件的，就是最近的，直接退出循环
                      # 如果没有满足条件的，则走完整个for循环
        dy_min = dy # for循环退出时的dy值就是实际情况中的最近距离
        for dy in range(dy_min+1, min(2*dy_min, dy_max)+1): # 这个for循环用于检查对称的一边是否有其它棋子
            x_new, y_new = p.x, p.y+dy # 命名为x_new,y_new以防止x,y的值被扔掉
            if (x_new, y_new) not in pDict.keys(): break # 超出棋盘，直接break
            if pDict[(x_new, y_new)].color != WHITE: # 这意味着对称的一边上有其它棋子
                fSet.discard((x,y)) # 这里的x,y还是上一个for循环退出时的值
                break # 加一个break提升性能
        for dy in range(1, dy_max+1): # 反方向来一遍
            x,y = p.x, p.y-dy # 加号变减号
            if (x,y) not in pDict.keys(): break
            if pDict[(x,y)].color != WHITE:
                fSet.add((x,y))
                break
        dy_min = dy # 反方向的dy_min和正方向的dy_min不一定相等
        for dy in range(dy_min+1, min(2*dy_min, dy_max)+1):
            x_new, y_new = p.x, p.y-dy
            if (x_new, y_new) not in pDict.keys(): break
            if pDict[(x_new, y_new)].color != WHITE:
                fSet.discard((x,y))
                break

        # 第二种情况: y坐标相等
        # 比较x的差值即可，正负分开，同上
        dx_max = 12
        for dx in range(1, dx_max+1):
            x,y = p.x+dx, p.y
            if (x,y) not in pDict.keys(): break
            if pDict[(x,y)].color != WHITE:
                fSet.add((x,y))
                break
        dx_min = dx
        for dx in range(dx_min+1, min(2*dx_min, dx_max)+1):
            x_new, y_new = p.x+dx, p.y
            if (x_new, y_new) not in pDict.keys(): break
            if pDict[(x_new, y_new)].color != WHITE:
                fSet.discard((x,y))
                break
        for dx in range(1, dx_max+1): # 反方向
            x,y = p.x-dx, p.y
            if (x,y) not in pDict.keys(): break
            if pDict[(x,y)].color != WHITE:
                fSet.add((x,y))
                break
        dx_min = dx
        for dx in range(dx_min+1, min(2*dx_min, dx_max)+1):
            x_new, y_new = p.x-dx, p.y
            if (x_new, y_new) not in pDict.keys(): break
            if pDict[(x_new, y_new)].color != WHITE:
                fSet.discard((x,y))
                break

        # 第三种情况：p.x - x = p.y - y
        d_max = 12
        for d in range(1, d_max+1):
            x,y = p.x+d, p.y+d
            if (x,y) not in pDict.keys(): break
            if pDict[(x,y)].color != WHITE:
                fSet.add((x,y))
                break
        d_min = d
        for d in range(d_min+1, min(2*d_min, d_max)+1):
            x_new, y_new = p.x+d, p.y+d
            if (x_new, y_new) not in pDict.keys(): break
            if pDict[(x_new, y_new)].color != WHITE:
                fSet.discard((x,y))
                break
        for d in range(1, d_max+1):
            x,y = p.x-d, p.y-d
            if (x,y) not in pDict.keys(): break
            if pDict[(x,y)].color != WHITE:
                fSet.add((x,y))
                break
        d_min = d
        for d in range(d_min+1, min(2*d_min, d_max)+1):
            x_new, y_new = p.x-d, p.y-d
            if (x_new, y_new) not in pDict.keys(): break
            if pDict[(x_new, y_new)].color != WHITE:
                fSet.discard((x,y))
                break

        return fSet
    def create_tpSet(p, fSet):
        # 生成所有可能的跳跃目标点tp
        # 注意：起点p和跳跃目标点tp关于f对称，把这个关系转换成向量即可
        # 'tp' stands for target points
        # 注意，当用于递归时，参数p不是起点p0
        tpSet = set()
        for f in fSet:
            vector = (f[0]-p.x, f[1]-p.y)
            tp = (f[0]+vector[0], f[1]+vector[1])
            tpSet.add(tp)
        return tpSet
    def delete_invalid_points(s, filter=False):
        # 删去s中非法的点，s是一个集合(set)
        # 非法的点包括：
        #     * 有棋子的点，所以要留住颜色为白色的点
        #     * 棋盘外的点
        #     * 特殊点，包括两类：
        #          * 起点p0
        #          * 已经算出的结果集vpSet，以避免无穷递归
        #          * 特殊点存放在specials中
        #          * 由于vpSet和tpSet都调用了这个函数，所以引入了参数filter以分别两种情况
        # 注意：specials中的点，必须以自建坐标系中的坐标表示
        # 注意：要先检测(x,y) in pDict.keys()，否则会引发KeyError
        if filter:
            nonlocal vpSet # 调用目前状态下的vpSet，而不是作为参数传入
            specials = set()
            specials.add((p0.x, p0.y)) # p0是特殊点
            specials = set.union(specials, vpSet) # 所有vp都是特殊点

            """
            这里之所以不用
            nonlocal vpSet
            spec = vpSet
            spec.add((p0.x, p0.y))
            是因为在这种情况下，spec和vpSet指向同一个集合，这样会改变vpSet本身
            """
        else:
            specials = set() # 空集
        return {(x,y) for (x,y) in s if (x,y) in pDict.keys() and pDict[(x,y)].color == WHITE and (x,y) not in specials}

    def recur(p):
        print('recur p', p.x, p.y)
        # 这个函数的递归逻辑之所在
        fSet = create_fSet(p)
        tpSet = create_tpSet(p, fSet)
        tpSet = delete_invalid_points(tpSet, filter=True)

        nonlocal vpSet, pcDict
        vpSet = vpSet.union(tpSet) # 在每次迭代中更新vpSet，这样才能保证delete_invalid_points中的nonlocal语句的有效性

        if not tpSet and pcDict.get(p.get_coord(), None) == None:
            # 如果tpSet是空集，并且p之前没有儿子(这是针对p0而言的，它的邻居已经是它的儿子了)，
            # 将不会进行下面的for循环
            # 即使如此，本次递归的p也是一个没有子的父
            # 这条语句使程序更健壮
            pcDict[p.get_coord()] = set()

        for (x,y) in list(tpSet):
            # 这里强行list一下tpSet，就是强行耗尽它的迭代器
            # 因为在每次迭代中会改变tpSet的值
            # 这里的做法是确保for循环迭代的是最初的tpSet

            # 更新父子关系
            try:
                pcDict[p.get_coord()].add((x,y))
                # 这个语句揭示了父子的逻辑，具体见函数make_traceDict
                # 注意：父子关系仅限于一代，隔代不算父子
            except KeyError:
                # 如果键没有对应的值，就新建有一个键值对
                pcDict[p.get_coord()] = {(x,y)}

            print_dict(pcDict, 'pcDict')

            tpSet = set.union(tpSet, recur(Point(x,y)))      # 通过递归得到tpSet

        return tpSet

    def update_traceDict(pcDict, traceDict):
        """
        通过分析传入的pcDict，更新traceDict，实际上是重新创建traceDict。
        这是一个无返回值函数，在原地修改traceDict。
        """
        # 初始化traceDict，为所有点的轨迹初始化，也就是创建一个空列表
        # 所有需要建立轨迹的点，就是pcDict中的所有键，也就是所有父
        # 递归尽头的父，它的子的集合是空集，但它依然是父
        traceDict.clear() # 清空字典，这里不能用traceDict = {}，因为我们要保持对原字典的引用，才能原地修改字典值

        childs = set()
        for parent in pcDict.keys():
            childs = childs.union(pcDict[parent])

        # 从子出发，为所有的子建立和起点p0的关系
        def relate(child):
            for parent in pcDict.keys():
                if child in pcDict[parent]: break # 找爸爸，可能不止一个爸爸，但只要找到一个就够了
                                                  # 这也就是众多可能路径中的一条
            if child == p0.get_coord(): # 递归终点
                return [p0.get_coord()]
            else: return relate(parent) + [child]

        for child in childs:
            # 制定每个儿子的轨迹
            traceDict[child] = relate(child)

        return # update_traceDict函数退出

    print('p0', p0.x, p0.y)
    # pcDict 是一个字典
    # pc stands for parent-child
    # 它记录了所有tp之间的父子关系，
    # 注意：父子仅限于单次移动，也就是一代，隔代的不算父子
    # 具体见recur, 此外，create_initial_vpSet中也更新了pcDict
    # 键为父，值为子的集合
    # recur函数会在每一次递归中更新pcDict
    pcDict = {}

    # 第一步：生成包含所有 合法 邻位 的点集
    vpSet = create_initial_vpSet()
    vpSet = delete_invalid_points(vpSet)

    # 第二步：生成tpSet
    tpSet = recur(p0) # 递归的起点是p0，但过程中会出现p0以外的p

    # 第三步：合并vpSet和tpSet
    vpSet = set.union(vpSet, tpSet)

    # 额外工作：更新traceDict
    update_traceDict(pcDict, traceDict)
    print_dict(traceDict, 'traceDict')

    return vpSet

def print_dict(d, dictname):
    print('='*16)
    print('%s:' % dictname)
    for (key, value) in d.items():
        print(key, value)
    print('='*16)

def is_valid_move(p, vpSet):
    """
    布尔函数，判断落子p的位置是否合法。
    vpSet通过create_vpSet创建。
    """
    return (p.x, p.y) in vpSet

def is_valid_pick(p, currentPlayer):
    """
    布尔函数，判断位置p的棋子能否被currentPlayer拾起。
    合法需要满足两个要求：
        * 位置p有子，即不是空白
        * 位置p子的颜色与currentPlayer对应的颜色相同
    合在一起，就是p.color == 当前玩家的颜色
    """
    return p.color == playerList[currentPlayer]['color']

def highlight_vps(pDict, vpSet, color=YELLOW, reverse=False):
    """
    把vpSet中的点以特定的颜色显示在pDict中。
    当reverse为True时，则将这些点的颜色还原为白色（只有没有棋子的点可以成为vp，而没有棋子意味着它是白色的）。
    """
    for (x,y) in vpSet:
        if reverse: pDict[(x,y)].color = WHITE
        else: pDict[(x,y)].color = color

def after_drop(x,y):
    """
    放回棋子或者落子后的处理。
    因为两种情况相似，为了代码重用，所以写了这个函数。
    x,y是落子点的坐标。
    """
    global currentColor, currentChequer, currentVpSet
                                        # 因为在拿起棋子时，棋子所在的位置变成了白色
    pDict[(x,y)].color = currentColor   # 所以在放下棋子时，要把它还原回原来的颜色
                                        # 或者，对于落子的情况，则要改变目标点的颜色
    highlight_vps(pDict, currentVpSet, reverse=True) # 还原被高光的落点

    currentColor = WHITE
    currentChequer = None
    currentVpSet = set() # 还原状态

    draw_board(pDict) # 重绘棋盘

def show_trace(p0, tp, delay=500):
    """
    落子后，显示从p0出发，依次跳跃到目标(x,y)的轨迹。
    p0是起点的元组坐标，tp是终点的元组坐标。

    这个函数在after_drop之前调用。
    """
    highlight_vps(pDict, currentVpSet, reverse=True) # 还原被高光的落点
                                                     # 这个还将在after_drop中调用
                                                     # 这里提前调用，是为了把棋盘上的黄色点（或者其他高光颜色）先行恢复成白色
    draw_board(pDict) # 调用这个函数的时候，currentChequer不为None，
                      # 因此画面上可能在像素坐标系之外飘着一个棋子
                      # 这里调用draw_board，就是把这个飘着的棋子擦掉
    traceList = currentTraceDict.get(tp, None) # 沿途经过的所有点的列表，包括起点和终点，具体见create_vpSet里的update_traceDict
    if not traceList:
        # 如果traceList是None，说明tp不在traceDict中
        # 说明它不是跳跃的目标点，而是相邻的目标点
        # 这类点的轨迹只需要包括起点和终点两个点即可
        traceList = [p0, tp]
    color = playerList[currentPlayer]['color'] # 当前玩家的颜色
    for (x,y) in traceList:
        Point(x,y).draw(screen, color) # 不能用pDict[(x,y)].draw()，因为我们不想改变这些点的固有颜色
        pygame.display.update()
        pygame.time.wait(delay)
        draw_board(pDict) # draw_board将会把p0和tp和正在绘制的轨迹点都绘制为白色，
                          # 所以要放在display.update之后，而不是之前
    return

def terminate():
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    screen = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
    pygame.display.set_caption('pyJump')

    FPSCLOCK = pygame.time.Clock()
    FPS = 30

    pDict = create_2player_pDict()
    playerList = create_2player_playerList(pDict)

    currentChequer = None # 现在正在拿着的棋子，没有时为None，有时为Point实例
    currentColor = WHITE # 记录currentChequer的颜色, 初始为WHITE
    currentVpSet = set() # 存储了当前拿起的棋子的所有合法落子点，默认为空集
    currentTraceDict = {} # 预先创建的空字典，记录了到达所有vp的轨迹
    currentPlayer = 0 # 记录当前行动的玩家，所有参与游戏的玩家都被放在playerList中，currentPlayer是这个列表的索引，语法糖

    draw_board(pDict) # 初始化棋盘
    while True:
        for event in pygame.event.get(QUIT): terminate()
        for event in pygame.event.get(MOUSEBUTTONUP):
            if currentChequer and event.button == 3:
                # 这句话是后加的，逻辑不是按顺序的
                # 之所以放在前面是为了更加灵敏地反应

                # 拿着子儿呢，同时点击鼠标右键：放下棋子
                print('drop')
                after_drop(currentChequer.x, currentChequer.y) # 把棋子放回原位

            for x,y in pDict.keys():
                # x,y 是自建坐标系的坐标
                p = pDict[(x,y)] # p是Point类的实例
                if p.rect.collidepoint(event.pos):
                    # 碰撞检测: 鼠标按键位置是棋盘上的某个点
                    if not currentChequer and event.button == 1 and \
                       is_valid_pick(p, currentPlayer): # p是碰撞发生的位置，检测p是否合法
                        # 没拿着子儿，同时点击鼠标左键，拾起
                            print('pick up')
                            currentChequer = p # 更新状态
                            currentColor = p.color # 拿起棋子，意味着原先放有棋子的位置变为白色
                            p.color = WHITE        # 如果原地放回棋子，可以使用after_drop(currentChequer.x,y)恢复原来的颜色
                                                   # currentColor保存了棋子的颜色，以便after_drop引用
                            currentVpSet = create_vpSet(currentChequer, pDict, currentTraceDict)
                            # 只需要在拿起棋子的时刻更新vpSet即可，尽可能减少调用create_vpSet函数，以提升性能
                            highlight_vps(pDict, currentVpSet) # 在拿起棋子的一瞬，就要做好显示落点的准备
                    elif currentChequer and event.button == 1 and \
                         is_valid_move(p, currentVpSet): # p是移动的目标点，检测p是否合法
                        # 拿着子儿呢，同时点击鼠标左键，落子
                        # 注意“放下棋子”和“落子”的区别，前者是放回原处，后者是放到新的地方
                        print('落子')
                        # 落子后，就轮到下一个玩家行动
                        # 由于下面的after_drop会重绘棋盘，为了改变棋盘的底色，
                        # 把对于currentPlayer的修改放在前面
                        show_trace((currentChequer.x, currentChequer.y), (x,y)) # show_trace调用了draw_board
                                                                                # draw_board调用了currentPlayer
                                                                                # 所以这条语句要放在修改currentPlayer之前
                        if currentPlayer == len(playerList)-1: currentPlayer = 0
                        else: currentPlayer += 1

                        currentVpSet.discard((x,y)) # 需要将(x,y)从vpSet中除去
                                                    # 否则函数after_drop在调用highlight的reverse时
                                                    # 会把目标点也变成白色，这不是我们所希望的
                        after_drop(x,y) # 注意: 这里的(x,y)是碰撞检测的结果，也就是新的落子点--目标点

        if currentChequer:
            # 拿着子儿呢
            p = currentChequer # syntactic sugar

            # 断言如下事实，这是在拿起子的瞬间完成的
            assert p.color == WHITE and currentColor != WHITE

            for event in pygame.event.get(MOUSEMOTION):
                # 只有在拿着棋子的时候，才需要处理MOUSEMOTION类事件
                draw_board(pDict)
                Point.draw(p, screen, currentColor, point=event.pos)
        for event in pygame.event.get():
            if event.type == KEYUP and event.key == K_ESCAPE: terminate()
        pygame.display.update()
        FPSCLOCK.tick(FPS)
