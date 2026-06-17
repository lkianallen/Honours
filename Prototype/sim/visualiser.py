import pygame
import math
import random
from collections import deque
from sim.simulator import Simulator
from sim.shardnode import ShardNode
from sim.transaction import Transaction

# ── Palette ────────────────────────────────────────────────────────────────
BG          = (245, 245, 242)
PANEL_BG    = (235, 234, 230)
BORDER      = (180, 178, 170)

AU_COL      = (52,  120, 190)   # blue
EU_COL      = (60,  170, 100)   # green
US_COL      = (210, 100,  60)   # orange

CPU_COL     = (90,  160, 230)
DISK_COL    = (230, 180,  60)
CPU_LOW     = (180, 215, 250)
DISK_LOW    = (250, 230, 160)

RELAY_COL   = (120, 100, 200)
PACKET_TRAIL= (200, 190, 230)

TEXT_DARK   = (30,  30,  30)
TEXT_MID    = (100, 100, 100)
TEXT_LIGHT  = (160, 160, 155)

VIOLATION   = (220,  50,  50)
OK_GREEN    = (60,  180,  80)

JURISDICTION_COLS = {b'1': AU_COL, b'2': EU_COL, b'3': US_COL}
JURISDICTION_NAMES = {b'1': 'AU  APP 8', b'2': 'EU  GDPR', b'3': 'US  Federal'}

# ── Layout constants ────────────────────────────────────────────────────────
W, H        = 1280, 800
PANEL_X     = 920
NODE_W, NODE_H = 72, 52
NODE_R      = 8          # border radius
RELAY_R     = 34

# ── Packet ──────────────────────────────────────────────────────────────────
class Packet:
    SPEED = 3.2

    def __init__(self, src, dst, colour, compliant=True):
        self.pos   = list(src)
        self.src   = list(src)
        self.dst   = list(dst)
        self.colour= colour
        self.compliant = compliant
        self.trail = deque(maxlen=18)
        self.done  = False
        dx = dst[0] - src[0]; dy = dst[1] - src[1]
        dist = math.hypot(dx, dy) or 1
        self.vel = [dx / dist * self.SPEED, dy / dist * self.SPEED]

    def update(self):
        self.trail.appendleft(tuple(self.pos))
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        if math.hypot(self.pos[0]-self.dst[0], self.pos[1]-self.dst[1]) < self.SPEED + 1:
            self.done = True

    def draw(self, surf):
        for i, p in enumerate(self.trail):
            alpha = int(200 * (1 - i / len(self.trail))) if self.trail else 0
            r = max(1, 4 - i // 4)
            col = PACKET_TRAIL if self.compliant else (240, 120, 120)
            s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*col, alpha), (r, r), r)
            surf.blit(s, (int(p[0])-r, int(p[1])-r))
        dot_col = self.colour if self.compliant else VIOLATION
        pygame.draw.circle(surf, dot_col, (int(self.pos[0]), int(self.pos[1])), 5)
        pygame.draw.circle(surf, (255,255,255), (int(self.pos[0]), int(self.pos[1])), 3)


# ── Node visual ─────────────────────────────────────────────────────────────
def draw_rounded_rect(surf, colour, rect, radius):
    pygame.draw.rect(surf, colour, rect, border_radius=radius)

def draw_node(surf, node_data, x, y, jur_col, fonts, selected=False):
    """
    node_data = {'id', 'cpu', 'disk', 'cpu_max', 'disk_max', 'queue'}
    Left half  = CPU fill   Right half = Disk fill
    """
    nw, nh = NODE_W, NODE_H
    rect = pygame.Rect(x, y, nw, nh)

    # shadow
    shadow = pygame.Rect(x+2, y+3, nw, nh)
    draw_rounded_rect(surf, (210, 208, 204), shadow, NODE_R)

    # background
    draw_rounded_rect(surf, (252, 252, 250), rect, NODE_R)

    cpu_frac  = min(1.0, 1.0 - node_data['cpu']  / node_data['cpu_max'])
    disk_frac = min(1.0, 1.0 - node_data['disk'] / node_data['disk_max'])

    half = nw // 2

    # CPU fill (left)
    if cpu_frac > 0:
        fill_h = int(nh * cpu_frac)
        fill_rect = pygame.Rect(x, y + nh - fill_h, half, fill_h)
        cpu_fill = CPU_COL if cpu_frac < 0.8 else VIOLATION
        pygame.draw.rect(surf, cpu_fill, fill_rect,
                         border_bottom_left_radius=NODE_R)

    # Disk fill (right)
    if disk_frac > 0:
        fill_h = int(nh * disk_frac)
        fill_rect = pygame.Rect(x + half, y + nh - fill_h, half, fill_h)
        disk_fill = DISK_COL if disk_frac < 0.8 else VIOLATION
        pygame.draw.rect(surf, disk_fill, fill_rect,
                         border_bottom_right_radius=NODE_R)

    # divider
    pygame.draw.line(surf, BORDER, (x + half, y + 6), (x + half, y + nh - 6), 1)

    # border
    border_col = jur_col if not selected else (50, 50, 50)
    draw_rounded_rect(surf, border_col, rect, NODE_R)
    pygame.draw.rect(surf, border_col, rect, width=2, border_radius=NODE_R)

    # label
    lbl = fonts['tiny'].render(f"N{node_data['id']}", True, TEXT_DARK)
    surf.blit(lbl, (x + nw//2 - lbl.get_width()//2, y + 4))

    # queue badge
    q = node_data['queue']
    if q > 0:
        badge_col = VIOLATION if q > 5 else (120, 120, 200)
        bx, by = x + nw - 14, y - 8
        pygame.draw.circle(surf, badge_col, (bx, by), 9)
        ql = fonts['tiny'].render(str(q), True, (255,255,255))
        surf.blit(ql, (bx - ql.get_width()//2, by - ql.get_height()//2))


def draw_relay(surf, cx, cy, fonts):
    pygame.draw.circle(surf, (220, 218, 215), (cx+2, cy+3), RELAY_R)
    pygame.draw.circle(surf, (250, 248, 245), (cx, cy), RELAY_R)
    pygame.draw.circle(surf, RELAY_COL, (cx, cy), RELAY_R, 2)
    lbl = fonts['tiny'].render('RELAY', True, RELAY_COL)
    surf.blit(lbl, (cx - lbl.get_width()//2, cy - lbl.get_height()//2))


def draw_zone_label(surf, text, x, y, col, font):
    lbl = font.render(text, True, col)
    surf.blit(lbl, (x - lbl.get_width()//2, y))


def draw_stats_panel(surf, stats, fonts, panel_x):
    pygame.draw.rect(surf, PANEL_BG, (panel_x, 0, W - panel_x, H))
    pygame.draw.line(surf, BORDER, (panel_x, 0), (panel_x, H), 1)

    pad = 18
    y = 20

    title = fonts['heading'].render('Simulation', True, TEXT_DARK)
    surf.blit(title, (panel_x + pad, y)); y += title.get_height() + 2
    sub = fonts['small'].render('Multi-Agent RL · Jurisdiction', True, TEXT_MID)
    surf.blit(sub, (panel_x + pad, y)); y += sub.get_height() + 18

    # divider
    pygame.draw.line(surf, BORDER, (panel_x + pad, y), (W - pad, y)); y += 12

    def stat_row(label, value, vcol=TEXT_DARK):
        lbl = fonts['small'].render(label, True, TEXT_MID)
        val = fonts['mono'].render(str(value), True, vcol)
        surf.blit(lbl, (panel_x + pad, y))
        surf.blit(val, (W - pad - val.get_width(), y))
        return lbl.get_height() + 6

    y += stat_row('Tick', stats['tick'])
    y += stat_row('Mode', stats['mode'])
    y += stat_row('TPS', f"{stats['tps']:.1f}")
    vcol = VIOLATION if stats['violations'] > 0 else OK_GREEN
    y += stat_row('Violations', stats['violations'], vcol)
    y += stat_row('In Flight', stats['in_flight'])
    y += 10

    pygame.draw.line(surf, BORDER, (panel_x + pad, y), (W - pad, y)); y += 14

    # per-jurisdiction queue totals
    jur_label = fonts['small'].render('Queue Depth by Zone', True, TEXT_MID)
    surf.blit(jur_label, (panel_x + pad, y)); y += jur_label.get_height() + 8

    for jur, col, name in [(b'1', AU_COL, 'AU'), (b'2', EU_COL, 'EU'), (b'3', US_COL, 'US')]:
        total = stats['queues'].get(jur, 0)
        bar_max = W - panel_x - pad*2
        bar_w = min(bar_max, int(bar_max * total / max(1, stats['max_queue'])))
        pygame.draw.rect(surf, (220, 218, 215), (panel_x + pad, y, bar_max, 12), border_radius=4)
        if bar_w > 0:
            pygame.draw.rect(surf, col, (panel_x + pad, y, bar_w, 12), border_radius=4)
        nl = fonts['tiny'].render(f'{name}  {total}', True, TEXT_DARK)
        surf.blit(nl, (panel_x + pad, y + 14))
        y += 32

    y += 8
    pygame.draw.line(surf, BORDER, (panel_x + pad, y), (W - pad, y)); y += 14

    # controls legend
    controls = [
        ('SPACE', 'Play / Pause'),
        ('→', 'Step one tick'),
        ('↑ ↓', 'Speed'),
        ('R', 'Reset'),
    ]
    ctrl_lbl = fonts['small'].render('Controls', True, TEXT_MID)
    surf.blit(ctrl_lbl, (panel_x + pad, y)); y += ctrl_lbl.get_height() + 6
    for key, desc in controls:
        k = fonts['mono'].render(key, True, RELAY_COL)
        d = fonts['tiny'].render(desc, True, TEXT_MID)
        surf.blit(k, (panel_x + pad, y))
        surf.blit(d, (panel_x + pad + 48, y))
        y += k.get_height() + 4

    y += 8
    speed_lbl = fonts['tiny'].render(f"Speed  ×{stats['speed']}", True, TEXT_MID)
    surf.blit(speed_lbl, (panel_x + pad, y))


# ── Main visualiser ──────────────────────────────────────────────────────────
class Visualiser:
    def __init__(self, simulator):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption('Blockchain Sharding · MAPPO Visualiser')
        self.clock  = pygame.time.Clock()
        self.sim    = simulator

        # fonts
        self.fonts = {
            'heading': pygame.font.SysFont('SF Pro Display, Helvetica Neue, Arial', 18, bold=True),
            'small'  : pygame.font.SysFont('SF Pro Text, Helvetica Neue, Arial', 13),
            'tiny'   : pygame.font.SysFont('SF Pro Text, Helvetica Neue, Arial', 11),
            'mono'   : pygame.font.SysFont('SF Mono, Menlo, Courier New', 13, bold=True),
        }

        self.packets   : list[Packet] = []
        self.playing   = False
        self.speed     = 1          # ticks per second when playing
        self.tick_acc  = 0.0
        self.violations= 0
        self.tps_history = deque(maxlen=20)

        self._build_layout()

    def _build_layout(self):
        # Relay sits in the centre-left area
        self.relay_pos = (PANEL_X // 2, H // 5)

        self.node_positions = {}  # node_id -> (x, y) top-left of node rect
        self.zone_centres   = {}  # jurisdiction bytes -> (cx, y) for label

        jurs = [b'1', b'2', b'3']
        num_zones = len(jurs)

        # Divide the canvas (excluding stats panel) into equal columns
        canvas_w = PANEL_X - 20
        col_w    = canvas_w // num_zones

        NODE_GAP  = 18   # vertical gap between nodes in a column
        LABEL_PAD = 32   # space above first node for zone label

        for col_idx, jur in enumerate(jurs):
            nodes_in_zone = [n for n in self.sim.nodes if n.jurisdiction == jur]
            num_nodes = len(nodes_in_zone)

            # Centre of this column
            cx = col_idx * col_w + col_w // 2

            # Total height of the node stack
            stack_h = num_nodes * NODE_H + (num_nodes - 1) * NODE_GAP

            # Start y so the stack is vertically centred, leaving room for label
            start_y = (H - stack_h) // 2 + LABEL_PAD // 2

            # Zone label sits above the stack
            self.zone_centres[jur] = (cx, start_y - LABEL_PAD)

            for row, node in enumerate(nodes_in_zone):
                nx = cx - NODE_W // 2
                ny = start_y + row * (NODE_H + NODE_GAP)
                self.node_positions[node.id] = (nx, ny)

    def _node_centre(self, node_id):
        x, y = self.node_positions[node_id]
        return (x + NODE_W // 2, y + NODE_H // 2)

    def spawn_packet(self, src_pos, dst_pos, jur, compliant=True):
        col = JURISDICTION_COLS.get(jur, (150, 150, 150))
        self.packets.append(Packet(src_pos, dst_pos, col, compliant))

    def _gather_stats(self):
        queues = {}
        max_q  = 1
        for node in self.sim.nodes:
            j = node.jurisdiction
            queues[j] = queues.get(j, 0) + len(node.write_queue)
            max_q = max(max_q, queues[j])
        return {
            'tick'      : self.sim.ticks,
            'mode'      : 'PLAYING' if self.playing else 'PAUSED',
            'tps'       : sum(self.tps_history) / max(1, len(self.tps_history)),
            'violations': self.violations,
            'in_flight' : len(self.packets),
            'queues'    : queues,
            'max_queue' : max_q,
            'speed'     : self.speed,
        }

    def step(self):
        """Advance one simulator tick and spawn packets."""
        before = {n.id: len(n.write_queue) for n in self.sim.nodes}
        self.sim.tick()
        after  = {n.id: len(n.write_queue) for n in self.sim.nodes}

        # packets: relay → nodes that gained transactions
        rx, ry = self.relay_pos
        processed = 0
        for node in self.sim.nodes:
            delta = after[node.id] - before[node.id]
            if delta > 0:
                for _ in range(delta):
                    self.spawn_packet(
                        (rx, ry),
                        self._node_centre(node.id),
                        node.jurisdiction
                    )
                    processed += 1
        self.tps_history.append(processed)

    def run(self, transactions):
        """Main loop.  Feed transactions into relay queue before starting."""
        self._pending_tx = list(transactions)
        self._tx_idx     = 0

        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            # ── events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        self._dispatch_next()
                        self.step()
                    elif event.key == pygame.K_UP:
                        self.speed = min(10, self.speed + 1)
                    elif event.key == pygame.K_DOWN:
                        self.speed = max(1, self.speed - 1)
                    elif event.key == pygame.K_r:
                        self._reset()

            # ── auto-advance ──
            queues_busy = any(len(n.write_queue) > 0 for n in self.sim.nodes)
            tx_remaining = self._tx_idx < len(self._pending_tx)
            if self.playing and (tx_remaining or queues_busy):
                self.tick_acc += dt * self.speed
                while self.tick_acc >= 1.0 and (tx_remaining or queues_busy):
                    if tx_remaining:
                        self._dispatch_next()
                        tx_remaining = self._tx_idx < len(self._pending_tx)
                    self.step()
                    queues_busy = any(len(n.write_queue) > 0 for n in self.sim.nodes)
                    self.tick_acc -= 1.0

            # ── update packets ──
            for p in self.packets:
                p.update()
            self.packets = [p for p in self.packets if not p.done]

            # ── draw ──
            self._draw()

        pygame.quit()

    def _dispatch_next(self):
        """Route next pending transaction and add to target node queue."""
        if self._tx_idx >= len(self._pending_tx):
            return
        tx = self._pending_tx[self._tx_idx]
        self._tx_idx += 1
        try:
            node_id = self.sim.route(tx)
            node    = next(n for n in self.sim.nodes if n.id == node_id)
            node.write_queue.append(tx)
        except ValueError:
            self.violations += 1

    def _reset(self):
        for node in self.sim.nodes:
            node.write_queue.clear()
            node.read_queue.clear()
            node.cpu  = node._cpu_max
            node.disk = node._disk_max
        self.sim.ticks   = 0
        self.packets     = []
        self.violations  = 0
        self.tps_history.clear()
        self._tx_idx     = 0
        self.playing     = False

    def _draw(self):
        self.screen.fill(BG)

        rx, ry = self.relay_pos

        # ── zone backgrounds ──
        for jur, col in JURISDICTION_COLS.items():
            zx, zy = self.zone_centres[jur]
            zone_surf = pygame.Surface((NODE_W*3 + 80, NODE_H*3 + 80), pygame.SRCALPHA)
            zone_surf.fill((*col, 18))
            self.screen.blit(zone_surf, (zx - NODE_W - 40, zy - 10))

        # ── relay lines ──
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        for jur in JURISDICTION_COLS:
            zx, zy_lbl = self.zone_centres[jur]
            zy = zy_lbl + RELAY_R + 10
            col = (*JURISDICTION_COLS[jur], 60)
            pygame.draw.line(s, col, (rx, ry), (zx, zy), 2)
        self.screen.blit(s, (0, 0))



        # ── relay node ──
        draw_relay(self.screen, rx, ry, self.fonts)

        # ── shard nodes ──
        for node in self.sim.nodes:
            x, y  = self.node_positions[node.id]
            jcol  = JURISDICTION_COLS.get(node.jurisdiction, (150,150,150))
            ndata = {
                'id'      : node.id,
                'cpu'     : node.cpu,
                'disk'    : node.disk,
                'cpu_max' : getattr(node, '_cpu_max', 100),
                'disk_max': getattr(node, '_disk_max', 1000),
                'queue'   : len(node.write_queue),
            }
            draw_node(self.screen, ndata, x, y, jcol, self.fonts)
        
        # ── packets (drawn after nodes so nodes sit below) ──
        for p in self.packets:
            p.draw(self.screen)

        # ── zone labels ──
        for jur, col in JURISDICTION_COLS.items():
            zx, zy = self.zone_centres[jur]
            name = JURISDICTION_NAMES[jur]
            draw_zone_label(self.screen, name, zx, zy, col, self.fonts['small'])

        # ── stats panel ──
        draw_stats_panel(self.screen, self._gather_stats(), self.fonts, PANEL_X)

        # ── tick counter watermark ──
        wm = self.fonts['heading'].render(f'TICK  {self.sim.ticks:04d}', True, (200, 198, 194))
        self.screen.blit(wm, (14, H - 30))

        pygame.display.flip()