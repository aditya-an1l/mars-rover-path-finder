import os
import time
import math
import heapq
import numpy as np
import pygame
import matplotlib as mpl
from math import sin, cos, sqrt, pi


class DEM:
    def __init__(self, nx=160, ny=160, scale=1.0):
        self.nx = nx
        self.ny = ny
        self.scale = scale
        self.z = np.zeros((nx, ny), dtype=float)
        self.slope_map = np.zeros_like(self.z)
        self.rough_map = np.ones_like(self.z)
        self.rocks = np.zeros((0, 3), dtype=float)

    def make_synthetic(self, bumps=12, noise=0.03, amplitude=2.0, seed=0):
        rng = np.random.RandomState(seed)
        xs = np.linspace(-1, 1, self.nx)
        ys = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.zeros_like(X)
        for _ in range(bumps):
            cx = rng.uniform(-0.8, 0.8)
            cy = rng.uniform(-0.8, 0.8)
            r = rng.uniform(0.05, 0.35)
            h = rng.uniform(-amplitude, amplitude)
            Z += h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r**2))
        Z += noise * rng.randn(*Z.shape)
        self.z = Z * (1.0 * self.scale)
        self.slope_map, self.rough_map = self.compute_slope_and_roughness()

    def compute_slope_and_roughness(self):
        dzdx = (np.roll(self.z, -1, axis=0) - np.roll(self.z, 1, axis=0)) / (
            2 * max(self.scale, 1e-9)
        )
        dzdy = (np.roll(self.z, -1, axis=1) - np.roll(self.z, 1, axis=1)) / (
            2 * max(self.scale, 1e-9)
        )
        grad = np.sqrt(dzdx**2 + dzdy**2)
        slope = np.degrees(np.arctan(grad))
        rough = 1.0 / np.cos(np.radians(np.clip(slope, -89.9, 89.9)))
        return slope, rough

    def place_procedural_rocks(self, n_rocks=25, min_r=1.0, max_r=3.2, seed=42):
        rng = np.random.RandomState(seed)
        rocks = []
        margin = 8
        for _ in range(n_rocks):
            i = rng.randint(margin, self.nx - margin)
            j = rng.randint(margin, self.ny - margin)
            r = rng.uniform(min_r, max_r)
            rocks.append([float(i), float(j), float(r)])
        self.rocks = np.array(rocks, dtype=float)

    def get_height(self, x, y):
        i = int(round(x))
        j = int(round(y))
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return 0.0
        return float(self.z[i, j])


def build_primitive_arc(curvature, length, n_samples=12):
    s = np.linspace(0, length, n_samples)
    theta = curvature * s
    ds = length / max(n_samples - 1, 1)
    xs = np.cumsum(np.cos(theta) * ds)
    ys = np.cumsum(np.sin(theta) * ds)
    xs = np.insert(xs[:-1], 0, 0.0)
    ys = np.insert(ys[:-1], 0, 0.0)
    headings = theta
    pts = np.vstack([xs, ys, headings]).T
    return pts


def generate_primitives_fast(
    lengths=[1.8, 3.0], radii=[np.inf, 6.0, 3.0], n_samples=12
):
    lib = []
    for L in lengths:
        for R in radii:
            k = 0.0 if np.isinf(R) else (1.0 / R)
            pts = build_primitive_arc(k, L, n_samples=n_samples)
            lib.append({"pts": pts, "length": L, "kappa": k})
    return lib


class MotionAStarFast:
    def __init__(self, dem, primitives, heading_bins=8, heuristic_weight=1.4):
        self.dem = dem
        self.primitives = primitives
        self.hbins = heading_bins
        self.heuristic_weight = heuristic_weight

        self.dzmax = 0.5
        self.phimax = 30.0
        self.rnmax = 3.5

        self._compute_cell_costs()

    def _compute_cell_costs(self):
        nx, ny = self.dem.nx, self.dem.ny
        self.cell_cost = np.zeros((nx, ny), dtype=np.float32)
        self.cell_ok = np.ones((nx, ny), dtype=np.bool_)
        z = self.dem.z
        slope = self.dem.slope_map
        rough = self.dem.rough_map
        rocks = getattr(self.dem, "rocks", None)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                center = z[i, j]
                neigh = z[i - 1 : i + 2, j - 1 : j + 2].flatten()
                maxdiff = float(np.max(np.abs(neigh - center)))
                if (
                    maxdiff > self.dzmax
                    or slope[i, j] > self.phimax
                    or rough[i, j] > self.rnmax
                ):
                    self.cell_ok[i, j] = False
                    self.cell_cost[i, j] = np.inf
                    continue
                c = (slope[i, j] / max(1.0, self.phimax)) ** 2 * 6.0 + (
                    (rough[i, j] - 1.0) / max(1.0, self.rnmax - 1.0)
                ) ** 2 * 4.0

                if rocks is not None and len(rocks) > 0:
                    dists = (
                        np.sqrt((rocks[:, 0] - i) ** 2 + (rocks[:, 1] - j) ** 2)
                        - rocks[:, 2]
                    )
                    min_signed = np.min(dists)
                    if min_signed < -0.4:
                        self.cell_ok[i, j] = False
                        self.cell_cost[i, j] = np.inf
                        continue
                    if min_signed < 5.0:
                        c += 3.5 * (1.0 / (max(1e-3, min_signed) + 1.0))
                self.cell_cost[i, j] = float(c)

        self.cell_ok[0, :] = False
        self.cell_ok[:, 0] = False
        self.cell_ok[-1, :] = False
        self.cell_ok[:, -1] = False

    def heuristic(self, x, y, gx, gy):
        return math.hypot(gx - x, gy - y)

    def sample_primitive_world(self, x0, y0, theta, prim):
        pts = prim["pts"]
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        xy = pts[:, 0:2] @ R.T + np.array([x0, y0])
        thetas = pts[:, 2] + theta
        return np.vstack([xy.T, thetas]).T

    def primitive_cost_by_cells(self, pts):
        nx, ny = self.dem.nx, self.dem.ny
        total = 0.0
        for p in pts:
            x, y, _ = p
            i = int(round(x))
            j = int(round(y))
            if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
                return np.inf
            if not self.cell_ok[i, j]:
                return np.inf
            total += self.cell_cost[i, j]
        return float(total)

    def plan(self, start, goal, max_iterations=20000, max_time=1.5):
        t0 = time.time()
        sx, sy, stheta = start
        gx, gy, _ = goal

        def theta_to_idx(th):
            thn = (th + 2 * pi) % (2 * pi)
            return int(round(thn / (2 * pi) * (self.hbins - 1))) % self.hbins

        start_key = (int(round(sx)), int(round(sy)), theta_to_idx(stheta))
        gscore = {start_key: 0.0}
        came_from = {}
        OPEN = []
        heapq.heappush(
            OPEN,
            (
                self.heuristic(sx, sy, gx, gy) * self.heuristic_weight,
                start_key,
                (sx, sy, stheta),
            ),
        )
        visited = set()
        it = 0

        while OPEN and it < max_iterations:
            it += 1
            if (it & 31) == 0 and (time.time() - t0) > max_time:
                return None
            fcur, node, pose = heapq.heappop(OPEN)
            if node in visited:
                continue
            visited.add(node)
            xi, yi, thi = node
            xw, yw = float(xi), float(yi)
            thw = pose[2]

            if math.hypot(xw - gx, yw - gy) < 2.5:
                cur = node
                path = []
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start_key)
                path.reverse()
                coords = [(float(n[0]), float(n[1])) for n in path]
                return coords

            for prim in self.primitives:
                pts_world = self.sample_primitive_world(xw, yw, thw, prim)
                tcost = self.primitive_cost_by_cells(pts_world)
                if tcost == np.inf:
                    continue

                kappa = abs(prim.get("kappa", 0.0))
                straight_pen = 0.0
                if kappa < 1e-6:
                    straight_pen = 0.6
                CJ = prim["length"] + straight_pen
                tentative_g = gscore.get(node, 1e9) + CJ + tcost
                x_end = pts_world[-1, 0]
                y_end = pts_world[-1, 1]
                th_end = pts_world[-1, 2]
                end_i, end_j = int(round(x_end)), int(round(y_end))
                if (
                    end_i < 0
                    or end_i >= self.dem.nx
                    or end_j < 0
                    or end_j >= self.dem.ny
                ):
                    continue
                end_node = (end_i, end_j, theta_to_idx(th_end))
                if tentative_g < gscore.get(end_node, 1e9):
                    gscore[end_node] = tentative_g
                    came_from[end_node] = node
                    h = self.heuristic(x_end, y_end, gx, gy)
                    f = tentative_g + self.heuristic_weight * h
                    heapq.heappush(OPEN, (f, end_node, (x_end, y_end, th_end)))
        return None


CELL_PIX = 4
GRID_W = 160
GRID_H = 160
SCREEN_W = GRID_W * CELL_PIX
SCREEN_H = GRID_H * CELL_PIX


def height_to_rgb(zgrid, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.min(zgrid))
    if vmax is None:
        vmax = float(np.max(zgrid))
    if vmax - vmin < 1e-9:
        norm = np.zeros_like(zgrid)
    else:
        norm = np.clip((zgrid - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = mpl.colormaps.get_cmap("terrain")
    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def create_terrain_surface_from_rgb(rgb, cell_pix):
    arr = np.transpose(rgb, (1, 0, 2)).copy()
    surf = pygame.surfarray.make_surface(arr)
    width = rgb.shape[0] * cell_pix
    height = rgb.shape[1] * cell_pix
    if (surf.get_width(), surf.get_height()) != (width, height):
        surf = pygame.transform.scale(surf, (width, height))
    return surf


def load_and_prepare_rover_image(desired_pixel_size):
    fname = "rover.png"
    if not os.path.isfile(fname):
        return None
    try:
        img = pygame.image.load(fname).convert_alpha()
    except Exception:
        return None
    w, h = img.get_width(), img.get_height()
    max_side = max(w, h)
    if max_side == 0:
        return None
    scale = float(desired_pixel_size) / float(max_side)
    base = pygame.transform.rotozoom(img, 0, scale)
    return base


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Rover control — fast planner")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    dem = DEM(nx=GRID_W, ny=GRID_H, scale=1.0)
    dem.make_synthetic(bumps=20, noise=0.02, amplitude=1.2, seed=1)
    dem.place_procedural_rocks(n_rocks=26, min_r=1.0, max_r=3.0, seed=7)
    rgb = height_to_rgb(dem.z)
    terrain_surf = create_terrain_surface_from_rgb(rgb, CELL_PIX)

    primitives = generate_primitives_fast(
        lengths=[1.8, 3.0], radii=[np.inf, 6.0, 3.0], n_samples=12
    )
    planner = MotionAStarFast(dem, primitives, heading_bins=8, heuristic_weight=1.4)

    rover = {"x": GRID_W * 0.2, "y": GRID_H * 0.2, "theta": 0.0}
    goal = None
    path = None
    auto_follow = False
    auto_target_idx = 0

    ROVER_PIXEL_SIZE = CELL_PIX * 6
    rover_base_img = load_and_prepare_rover_image(ROVER_PIXEL_SIZE)
    use_rover_image = rover_base_img is not None
    if use_rover_image:
        print("Loaded rover.png (sprite mode).")
    else:
        print("No rover.png found -> using triangle fallback.")

    instructions = [
        "Drive: arrows or WASD (forward/back, rotate)",
        "Left-click: set goal",
        "'p' : plan path from rover to goal (fast)",
        "'f' : toggle follow path",
        "'r' : regenerate terrain",
        "'Esc' or 'q' : quit",
    ]

    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_r:
                    dem.make_synthetic(
                        bumps=18,
                        noise=0.03,
                        amplitude=1.2,
                        seed=np.random.randint(0, 10000),
                    )
                    dem.place_procedural_rocks(
                        n_rocks=26,
                        min_r=1.0,
                        max_r=3.0,
                        seed=np.random.randint(0, 10000),
                    )
                    rgb = height_to_rgb(dem.z)
                    terrain_surf = create_terrain_surface_from_rgb(rgb, CELL_PIX)

                    primitives = generate_primitives_fast(
                        lengths=[1.8, 3.0], radii=[np.inf, 6.0, 3.0], n_samples=12
                    )
                    planner = MotionAStarFast(
                        dem, primitives, heading_bins=8, heuristic_weight=1.4
                    )
                    path = None
                    goal = None
                    auto_follow = False
                elif ev.key == pygame.K_p:
                    if goal is not None:
                        start = (rover["x"], rover["y"], rover["theta"])
                        goal_pose = (goal[0], goal[1], 0.0)
                        print("Planning path (fast)...")
                        t0 = time.time()
                        path_nodes = planner.plan(
                            start, goal_pose, max_iterations=30000, max_time=1.5
                        )
                        t_elapsed = time.time() - t0
                        if path_nodes is None:
                            print("Planner failed / timed out (%.2fs)." % (t_elapsed,))
                            path = None
                        else:
                            path = np.array(path_nodes)
                            print(
                                "Path found: %d nodes (%.2fs)" % (len(path), t_elapsed)
                            )
                            auto_target_idx = 1
                    else:
                        print("No goal set.")
                elif ev.key == pygame.K_f:
                    if path is None:
                        print("No path to follow.")
                    else:
                        auto_follow = not auto_follow
                        print("Auto-follow:", auto_follow)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    mx, my = ev.pos
                    gx = mx / CELL_PIX
                    gy = my / CELL_PIX
                    gx = max(0, min(GRID_W - 1, gx))
                    gy = max(0, min(GRID_H - 1, gy))
                    goal = (gx, gy)
                    path = None
                    auto_follow = False
                    print("Goal set to: (%.2f, %.2f)" % (gx, gy))

        pressed = pygame.key.get_pressed()
        forward = 0.0
        rotate = 0.0
        speed = 6.0 * dt
        rot_speed = 2.0 * dt
        if pressed[pygame.K_UP] or pressed[pygame.K_w]:
            forward = speed
        if pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
            forward = -speed * 0.6
        if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
            rotate = -rot_speed
        if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
            rotate = rot_speed

        if not auto_follow:
            rover["theta"] += rotate
            rover["x"] += forward * math.cos(rover["theta"])
            rover["y"] += forward * math.sin(rover["theta"])
        else:
            if path is not None and auto_target_idx < len(path):
                target = path[auto_target_idx]
                tx, ty = float(target[0]), float(target[1])
                dx = tx - rover["x"]
                dy = ty - rover["y"]
                dist = math.hypot(dx, dy)
                if dist < 0.8:
                    auto_target_idx += 1
                else:
                    desired_theta = math.atan2(dy, dx)
                    dtheta = (desired_theta - rover["theta"] + pi) % (2 * pi) - pi
                    max_rot = 0.06
                    dtheta = max(-max_rot, min(max_rot, dtheta))
                    rover["theta"] += dtheta
                    step = 0.6
                    rover["x"] += step * math.cos(rover["theta"]) * dt * 6.0
                    rover["y"] += step * math.sin(rover["theta"]) * dt * 6.0

        rover["x"] = max(0.0, min(GRID_W - 1, rover["x"]))
        rover["y"] = max(0.0, min(GRID_H - 1, rover["y"]))

        screen.blit(terrain_surf, (0, 0))

        if hasattr(dem, "rocks") and dem.rocks is not None and len(dem.rocks) > 0:
            for rx, ry, rr in dem.rocks:
                pygame.draw.circle(
                    screen,
                    (30, 30, 30),
                    (int(rx * CELL_PIX), int(ry * CELL_PIX)),
                    max(2, int(rr * CELL_PIX)),
                )
                pygame.draw.circle(
                    screen,
                    (80, 80, 80),
                    (int(rx * CELL_PIX), int(ry * CELL_PIX)),
                    max(1, int(rr * CELL_PIX)),
                    1,
                )

        if path is not None and len(path) > 0:
            pts = [(int(p[0] * CELL_PIX), int(p[1] * CELL_PIX)) for p in path]
            pygame.draw.lines(screen, (255, 200, 0), False, pts, 3)
            for p in pts:
                pygame.draw.circle(screen, (255, 220, 80), p, 3)

        rx = int(rover["x"] * CELL_PIX)
        ry = int(rover["y"] * CELL_PIX)
        heading = rover["theta"]
        if use_rover_image and rover_base_img is not None:
            deg = -math.degrees(heading)
            rotated = pygame.transform.rotozoom(rover_base_img, deg, 1.0)
            rrect = rotated.get_rect(center=(rx, ry))
            screen.blit(rotated, rrect.topleft)
        else:
            L = 8
            p1 = (rx + int(L * math.cos(heading)), ry + int(L * math.sin(heading)))
            p2 = (
                rx + int(L * math.cos(heading + 2.2)),
                ry + int(L * math.sin(heading + 2.2)),
            )
            p3 = (
                rx + int(L * math.cos(heading - 2.2)),
                ry + int(L * math.sin(heading - 2.2)),
            )
            pygame.draw.polygon(screen, (0, 255, 0), [p1, p2, p3])
            pygame.draw.circle(screen, (0, 0, 0), (rx, ry), 3)

        if goal is not None:
            gx = int(goal[0] * CELL_PIX)
            gy = int(goal[1] * CELL_PIX)
            pygame.draw.circle(screen, (255, 50, 50), (gx, gy), 6)
            pygame.draw.circle(screen, (255, 150, 150), (gx, gy), 3)

        lines = [
            "Pos: (%.1f, %.1f)  Heading: %.1f deg"
            % (rover["x"], rover["y"], math.degrees(rover["theta"]) % 360),
            "Goal: %s"
            % ("None" if goal is None else "(%.1f, %.1f)" % (goal[0], goal[1])),
            "Path: %d nodes" % (0 if path is None else len(path)),
            "Auto-follow: %s" % auto_follow,
        ]
        y0 = 4
        for ln in lines:
            txt = font.render(ln, True, (255, 255, 255))
            screen.blit(txt, (6, y0))
            y0 += 18

        y0 += 6
        for inst in instructions:
            txt = font.render(inst, True, (220, 220, 220))
            screen.blit(txt, (6, y0))
            y0 += 16

        pygame.display.flip()

    pygame.quit()
    print("Exited.")


if __name__ == "__main__":
    main()
