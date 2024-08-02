# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import warp as wp
import math

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

wp.config.verify_fp = True
wp.init()


@wp.struct
class RenderMesh:
    """Mesh to be ray traced.
    Assumes a triangle mesh as input.
    Per-vertex normals are computed with compute_vertex_normals()
    """

    id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    pos: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.quat)


@wp.struct
class Camera:
    """Basic camera for ray tracing"""

    horizontal: float
    vertical: float
    aspect: float
    tan: float
    pos: wp.vec3
    rot: wp.quat


class Example:
    """A basic differentiable ray tracer

    Non-differentiable variables:
    camera.horizontal: camera horizontal aperture size
    camera.vertical: camera vertical aperture size
    camera.aspect: camera aspect ratio
    camera.pos: camera displacement
    camera.rot: camera rotation (quaternion)
    pix_width: final image width in pixels
    pix_height: final image height in pixels
    render_mesh.indices: mesh vertex indices

    Differentiable variables:
    render_mesh.pos: parent transform displacement
    render_mesh.quat: parent transform rotation (quaternion)
    render_mesh.vertices: mesh vertex positions

    Note that: You are not supposed to change the name of this class!!!
    It should be kept as 'Example'.
    Otherwise, the render() method will raise an exception about not finding the kernel method.
    """

    def __init__(self, points_, indices_, device):
        cam_pos = wp.vec3(37, 37, 1)
        cam_rot = wp.quat(0.707, 0.0, 0.0, 0.707)
        self.device = device

        horizontal_aperture = 106  # Realsense resolution
        vertical_aperture = 60  # Realsense resolution
        aspect = vertical_aperture / horizontal_aperture

        self.width = 106
        self.height = int(aspect * self.width)
        self.num_pixels = self.width * self.height
        self.cam_pos_terrain = []

        points = points_
        indices = indices_

        with wp.ScopedDevice(device=self.device):
            # construct RenderMesh
            self.render_mesh = RenderMesh()
            self.mesh = wp.Mesh(
                points=wp.array(points, dtype=wp.vec3, requires_grad=False), indices=wp.array(indices, dtype=int)
            )
            self.render_mesh.id = self.mesh.id
            self.render_mesh.vertices = self.mesh.points
            self.render_mesh.indices = self.mesh.indices
            self.render_mesh.pos = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
            self.render_mesh.rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0]), dtype=wp.quat, requires_grad=False)

            # construct camera
            self.camera = Camera()
            self.camera.horizontal = horizontal_aperture
            self.camera.vertical = vertical_aperture
            self.camera.aspect = aspect
            self.camera.tan = np.tan(np.radians(87 / 2))  # FOV
            self.camera.pos = cam_pos
            self.camera.rot = cam_rot

            self.depth = wp.zeros(self.num_pixels, dtype=float, requires_grad=False)

    def update_cam_pos(self, cam_pos, cam_rot, cam_tan, env_pos=None, border_size=None):
        """

        :param cam_pos: world frame position
        :param cam_rot: world frame orientation
        :param cam_tan: tangent of camera fov * 0.5
        :param env_pos:
        :param border_size:
        :return:
        """
        # if border_size is None:
        #     border_size = [10, 10]
        self.cam_pos_terrain = [cam_pos.x, cam_pos.y, cam_pos.z]
        # if env_pos is not None:
        #     self.cam_pos_terrain[0] += env_pos[0]
        #     self.cam_pos_terrain[1] += env_pos[1]
        #     self.cam_pos_terrain[2] += env_pos[2]

        self.camera.pos = wp.vec3(self.cam_pos_terrain)
        self.camera.tan = cam_tan
        self.camera.rot = wp.quat(cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w)

    def update(self):
        pass

    def render(self):
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=Example.draw_kernel,
                dim=self.num_pixels,
                inputs=[
                    self.render_mesh,
                    self.camera,
                    self.width,
                    self.height,
                    self.depth,
                ]
            )

    @wp.kernel
    def draw_kernel(
            mesh: RenderMesh,
            camera: Camera,
            rays_width: int,
            rays_height: int,
            depth: wp.array(dtype=float),
    ):
        tid = wp.tid()

        x = tid % rays_width
        y = rays_height - tid // rays_width

        # compute standard coordinate
        sx = 2.0 * float(x) / float(rays_width) - 1.0
        sy = 2.0 * float(y) / float(rays_height) - 1.0

        # compute view ray in world space
        ro_world = camera.pos
        rd_world = wp.normalize(
            wp.quat_rotate(camera.rot, wp.vec3(sx * camera.tan, sy * camera.tan * camera.aspect, -1.0))
        )

        ry = math.atan(sy * camera.tan * camera.aspect)
        rx = math.atan(sx * camera.tan)

        # compute view ray in mesh space
        inv = wp.transform_inverse(wp.transform(mesh.pos[0], mesh.rot[0]))
        ro = wp.transform_point(inv, ro_world)  # ray origin
        rd = wp.transform_vector(inv, rd_world)  # ray directions

        t = float(0.0)
        ur = float(0.0)
        vr = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        if wp.mesh_query_ray(mesh.id, ro, rd, 2.1, t, ur, vr, sign, n, f):
            dis_ = t * math.cos(ry) * math.cos(rx)

        if dis_ >= 2. or dis_ <= 0.000001:
            depth[tid] = 2.
        elif dis_ <= 0.2:
            depth[tid] = 0.2
        else:
            depth[tid] = dis_

    def get_depth_map(self):
        depth_map_ = wp.torch.to_torch(self.depth.reshape((self.height, self.width)))
        return depth_map_


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots, device) -> None:

        self.device = device
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        self.has_depth_sensor = True
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        if len(cfg.terrain_proportions) < 9:
            cfg.terrain_proportions = cfg.terrain_proportions + [0] * (9-len(cfg.terrain_proportions))
        cfg.terrain_proportions = np.array(cfg.terrain_proportions)
        if np.sum(cfg.terrain_proportions) > 1:
            cfg.terrain_proportions /= np.sum(cfg.terrain_proportions)
        elif np.sum(cfg.terrain_proportions) < 1:
            cfg.terrain_proportions[-1] = np.sum(cfg.terrain_proportions[:-1])
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border_length = int(cfg.border_length / self.cfg.horizontal_scale)
        self.border_width = int(cfg.border_width / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border_width
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border_length
        min_height = int(self.cfg.heightfeild_range[0] / self.cfg.vertical_scale)
        max_height = int(self.cfg.heightfeild_range[1] / self.cfg.vertical_scale)
        step = int(self.cfg.heightfeild_resolution / self.cfg.vertical_scale)
        heights_range = np.arange(min_height, max_height + step, step)
        self.height_field_raw = np.random.choice(heights_range, (self.tot_rows, self.tot_cols))
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        # if self.type == "trimesh" or self.type == "heightfeild":
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                     self.cfg.horizontal_scale,
                                                                                     self.cfg.vertical_scale,
                                                                                     self.cfg.slope_treshold)

        points = self.vertices
        indices = self.triangles.flatten(order='C')
        if self.has_depth_sensor == True:
            self.example = Example(points, indices, self.device)
        else:
            self.example = None

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = 0.9
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.vertical_scale,
                                               horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.length_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.6
        step_height = 0.05 + 0.15 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5
        stone_distance = 0.5 * (1.05 - difficulty)
        gap_size = 0.9 * difficulty + 0.1
        pit_depth = 0.1 + 0.75 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            if choice < self.proportions[1] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfeild_range[0] + self.cfg.rough_slope_range[0],
                                                 max_height=self.cfg.heightfeild_range[1] + self.cfg.rough_slope_range[1],
                                                 step=self.cfg.heightfeild_resolution,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice > self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=np.random.uniform(0.31, 0.51),
                                                 step_height=step_height, platform_size=3)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfeild_range[0],
                                                 max_height=self.cfg.heightfeild_range[1],
                                                 step=self.cfg.heightfeild_resolution,
                                                 downsampled_scale=0.1)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain,
                                                  stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance,
                                                  max_height=0.,
                                                  platform_size=4.)
        elif choice < self.proportions[6]:
            terrain_utils.gap_terrain(terrain, gap_size=gap_size)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfeild_range[0],
                                                 max_height=self.cfg.heightfeild_range[1],
                                                 step=self.cfg.heightfeild_resolution,
                                                 downsampled_scale=0.1)
        elif choice < self.proportions[7]:
            terrain_utils.pit_terrain(terrain, depth=pit_depth)
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfeild_range[0],
                                                 max_height=self.cfg.heightfeild_range[1],
                                                 step=self.cfg.heightfeild_resolution,
                                                 downsampled_scale=0.1)
        else:
            terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=self.cfg.heightfeild_range[0],
                                                 max_height=self.cfg.heightfeild_range[1],
                                                 step=self.cfg.heightfeild_resolution,
                                                 downsampled_scale=0.1)
        terrain.transpose()
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border_length + i * self.length_per_env_pixels
        end_x = self.border_length + (i + 1) * self.length_per_env_pixels
        start_y = self.border_width + j * self.width_per_env_pixels
        end_y = self.border_width + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 0.5) / terrain.horizontal_scale)
        # print("region: ",x1, x2, y1, y2)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
