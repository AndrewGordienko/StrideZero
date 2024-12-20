__credits__ = ["Andrea PIERRÉ"]

import math
import numpy as np
from typing import Optional, List

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'Box2D is not installed. Install it via `pip install "gymnasium[box2d]"`.'
    ) from e

FPS = 50
SCALE = 30.0
MOTORS_TORQUE = 150
SPEED_HIP = 3
SPEED_KNEE = 4.5
LIDAR_RANGE = 160 / SCALE
INITIAL_RANDOM = 5
HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
SCALE_FACTOR = 4 / 3
HULL_POLY = [(x * SCALE_FACTOR, y * SCALE_FACTOR) for x, y in HULL_POLY]

LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE
VIEWPORT_W = 600
VIEWPORT_H = 400
TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10
TERRAIN_STARTPAD = 20
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,
    restitution=0.0,
)

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.hull == contact.fixtureA.body
            or self.env.hull == contact.fixtureB.body
        ):
            self.env.game_over = True
        for leg_part in self.env.ground_contacts:
            if leg_part in [contact.fixtureA.body, contact.fixtureB.body]:
                leg_part.ground_contact = True

    def EndContact(self, contact):
        for leg_part in self.env.ground_contacts:
            if leg_part in [contact.fixtureA.body, contact.fixtureB.body]:
                leg_part.ground_contact = False


class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
        EzPickle.__init__(self, render_mode, hardcore)
        self.world = Box2D.b2World()
        self.terrain = []
        self.hull = None
        self.prev_shaping = None
        self.hardcore = hardcore
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )
        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # Action space: 6 actions (up to 3 motors per leg)
        self.action_space = spaces.Box(
            np.array([-1.0] * 6, dtype=np.float32),
            np.array([1.0] * 6, dtype=np.float32),
        )

        # Observation space: Hull info + 2 legs * (3 joints * 2 + ground contact) + 10 lidar readings
        low = np.array(
            [-math.pi, -5.0, -5.0, -5.0] + [-math.pi, -5.0] * 3 * 2 + [0.0] * 2 + [-1.0] * 10,
            dtype=np.float32,
        )
        high = np.array(
            [math.pi, 5.0, 5.0, 5.0] + [math.pi, 5.0] * 3 * 2 + [1.0] * 2 + [1.0] * 10,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low, high)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        if self.hull:
            self.world.DestroyBody(self.hull)
            self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        stair_steps, stair_width, stair_height = 0, 0, 0
        original_y = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.random() > 0.5 else -1
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (
                    x
                    + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                    y
                    + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                )
                for a in range(5)
            ]
            x1 = min(p[0] for p in poly)
            x2 = max(p[0] for p in poly)
            self.cloud_poly.append((poly, x1, x2))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener = ContactDetector(self)
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H + 1.5
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        self.legs = []
        self.joints = []
        self.num_subparts_list = []
        self.motor_joints = []
        self.ground_contacts = []
        joint_idx = 0

        for i in [-1, +1]:
            # num_subparts = self.np_random.choice([2, 3])
            num_subparts = 3
            self.num_subparts_list.append(num_subparts)
            # Upper leg
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            leg.ground_contact = False
            self.ground_contacts.append(leg)

            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=0.0,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))
            self.motor_joints.append(joint_idx)
            joint_idx += 1

            # Lower leg
            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=0.0,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.ground_contacts.append(lower)
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))
            self.motor_joints.append(joint_idx)
            joint_idx += 1

            # Foot
            # Adjust foot position based on the number of subparts
            if num_subparts == 3:
                foot_y = init_y - LEG_H * 5 / 2 - LEG_DOWN
            else:
                foot_y = init_y - LEG_H * 3 / 2 - LEG_DOWN

            # Foot
            foot = self.world.CreateDynamicBody(
                position=(init_x, foot_y),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )

            foot.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            foot.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            foot.ground_contact = False
            self.ground_contacts.append(foot)
            self.legs.append(foot)
            enable_motor = True if num_subparts == 3 else False
            rjd = revoluteJointDef(
                bodyA=lower,
                bodyB=foot,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=enable_motor,
                enableLimit=enable_motor,
                maxMotorTorque=MOTORS_TORQUE if enable_motor else 0.0,
                motorSpeed=0.0,
                lowerAngle=-1.6 if enable_motor else 0.0,
                upperAngle=-0.1 if enable_motor else 0.0,
            )
            self.joints.append(self.world.CreateJoint(rjd))
            if enable_motor:
                self.motor_joints.append(joint_idx)
                joint_idx += 1

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0.0] * 6))[0], {}

    def step(self, action: np.ndarray):
        assert self.hull is not None
        # Process actions for motorized joints
        assert len(action) == 6
        for idx, joint_idx in enumerate(self.motor_joints):
            self.joints[joint_idx].motorSpeed = float(
                SPEED_HIP * np.sign(action[idx])
            )
            self.joints[joint_idx].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[idx]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
        ]

        state = [
            # Hull state information
            self.hull.angle,  # Hull angle
            2.0 * self.hull.angularVelocity / FPS,  # Angular velocity
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Horizontal velocity
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,  # Vertical velocity

            # Joint and leg state information for Leg 0 (left leg)
            self.joints[0].angle,  # Hip angle for left leg
            self.joints[0].speed / SPEED_HIP,  # Hip speed for left leg
            self.joints[1].angle + 1.0,  # Knee angle for left leg
            self.joints[1].speed / SPEED_KNEE,  # Knee speed for left leg
            self.joints[2].angle,  # Lower joint angle for left leg (new subpart)
            self.joints[2].speed / SPEED_HIP,  # Lower joint speed for left leg (new subpart)
            1.0 if self.legs[0].ground_contact else 0.0,  # Ground contact for left leg (foot/ankle)

            # Joint and leg state information for Leg 1 (right leg)
            self.joints[3].angle,  # Hip angle for right leg
            self.joints[3].speed / SPEED_HIP,  # Hip speed for right leg
            self.joints[4].angle + 1.0,  # Knee angle for right leg
            self.joints[4].speed / SPEED_KNEE,  # Knee speed for right leg
            self.joints[5].angle,  # Lower joint angle for right leg (new subpart)
            self.joints[5].speed / SPEED_HIP,  # Lower joint speed for right leg (new subpart)
            1.0 if self.legs[1].ground_contact else 0.0,  # Ground contact for right leg (foot/ankle)
        ]

        leg_idx = 0
        joint_idx = 0
        # Loop to add exactly three joints (subparts) per leg without padding
        for leg_idx in range(2):  # Two legs
            for j in range(3):  # Three subparts per leg
                joint = self.joints[joint_idx]
                state.append(joint.angle)
                state.append(joint.speed / SPEED_HIP)
                joint_idx += 1
            
            # Ground contact for lower leg and foot
            lower_leg = self.ground_contacts[leg_idx * 2]
            foot = self.ground_contacts[leg_idx * 2 + 1]
            ground_contact = 1.0 if lower_leg.ground_contact or foot.ground_contact else 0.0
            state.append(ground_contact)

        state += [l.fraction for l in self.lidar]

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5
        shaping = 130 * pos[0] / SCALE - 6.0 * abs(state[0])
        reward = 0.0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00025 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, single_lidar.p1[1] * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, single_lidar.p2[1] * SCALE),
                    width=1,
                )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        flagy1 = TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        x = TERRAIN_STEP * 3 * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class BipedalWalkerHeuristics:
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    a = np.array([0.0, 0.0, 0.0, 0.0])

    def step_heuristic(self, s):
        moving_s_base = 4 + 5 * self.moving_leg
        supporting_s_base = 4 + 5 * self.supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if self.state == self.STAY_ON_ONE_LEG:
            hip_targ[self.moving_leg] = 1.1
            knee_targ[self.moving_leg] = -0.6
            self.supporting_knee_angle += 0.03
            if s[2] > self.SPEED:
                self.supporting_knee_angle += 0.03
            self.supporting_knee_angle = min(
                self.supporting_knee_angle, self.SUPPORT_KNEE_ANGLE
            )
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                self.state = self.PUT_OTHER_DOWN
        if self.state == self.PUT_OTHER_DOWN:
            hip_targ[self.moving_leg] = +0.1
            knee_targ[self.moving_leg] = self.SUPPORT_KNEE_ANGLE
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[moving_s_base + 4]:
                self.state = self.PUSH_OFF
                self.supporting_knee_angle = min(
                    s[moving_s_base + 2], self.SUPPORT_KNEE_ANGLE
                )
        if self.state == self.PUSH_OFF:
            knee_targ[self.moving_leg] = self.supporting_knee_angle
            knee_targ[self.supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * self.SPEED:
                self.state = self.STAY_ON_ONE_LEG
                self.moving_leg = 1 - self.moving_leg
                self.supporting_leg = 1 - self.moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        self.a[0] = hip_todo[0]
        self.a[1] = knee_todo[0]
        self.a[2] = hip_todo[1]
        self.a[3] = knee_todo[1]
        self.a = np.clip(0.5 * self.a, -1.0, 1.0)

        return self.a

