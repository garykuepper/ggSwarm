"""
tron_env.py — Cinematic Tron environment for ggSwarm showcase.

Sets up:
  - Pure black void stage
  - Volumetric fog atmosphere
  - Emissive blue grid ground plane
  - Blue emissive drone material override
  - Cinematic camera rigs (orbit, top-down, low-angle)
  - NVENC video recorder

Usage in play.py:
    from tron_env import setup_tron_environment, TronCameraRig
    setup_tron_environment(sim)
    cam = TronCameraRig(sim, mode="orbit")   # or "top_down", "low_angle"

Requires: Isaac Sim 5.1 / Isaac Lab 2.3
"""

import math
import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Gf, Sdf
import carb


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_tron_environment(sim, drone_prim_paths: list[str] | None = None):
    """
    Call once after sim.reset() in your play.py before the render loop.

    Args:
        sim:               The Isaac Lab SimulationContext instance.
        drone_prim_paths:  List of USD prim paths for each drone root.
                           If provided, applies blue emissive material to all.
                           Example: ["/World/envs/env_0/Robot_0", ...]
    """
    stage = omni.usd.get_context().get_stage()

    _set_black_void(stage)
    _add_fog(stage)
    _add_grid_plane(stage)
    _add_cinematic_lighting(stage)

    if drone_prim_paths:
        mat_path = _create_blue_emissive_material(stage)
        for prim_path in drone_prim_paths:
            _apply_material_to_drone(stage, prim_path, mat_path)

    carb.log_info("[tron_env] Tron environment setup complete.")


# ---------------------------------------------------------------------------
# Stage — black void
# ---------------------------------------------------------------------------

def _set_black_void(stage: Usd.Stage):
    """Remove default sky dome and set background to pure black."""
    # Kill the default distant light and dome if they exist
    for path in ["/World/defaultDistantLight", "/World/defaultDomeLight", "/Environment"]:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)

    # Set viewport background color to black via render settings
    settings = carb.settings.get_settings()
    settings.set("/rtx/backgroundColorR", 0.0)
    settings.set("/rtx/backgroundColorG", 0.0)
    settings.set("/rtx/backgroundColorB", 0.0)

    # Enable RTX path tracing for cinematic quality
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", 32)          # samples per pixel — raise for final render
    settings.set("/rtx/pathtracing/totalSpp", 128)
    carb.log_info("[tron_env] Black void stage set, RTX path tracing enabled.")


# ---------------------------------------------------------------------------
# Atmosphere — volumetric fog
# ---------------------------------------------------------------------------

def _add_fog(stage: Usd.Stage):
    """Add volumetric atmosphere fog for the smoky Tron look."""
    fog_path = "/World/TronFog"
    fog_prim = stage.DefinePrim(fog_path, "RectLight")  # placeholder — use render settings fog

    # Isaac Sim fog via render settings (simpler than USD volume)
    settings = carb.settings.get_settings()
    settings.set("/rtx/fog/enabled", True)
    # Fog tinted toward #64d5d2 (Gary Gigabytes signature teal)
    settings.set("/rtx/fog/fogColor/r", 0.02)
    settings.set("/rtx/fog/fogColor/g", 0.08)
    settings.set("/rtx/fog/fogColor/b", 0.08)
    settings.set("/rtx/fog/fogStartDistance", 3.0)
    settings.set("/rtx/fog/fogEndDistance", 25.0)
    settings.set("/rtx/fog/fogDensity", 0.08)    # subtle — drones still visible
    carb.log_info("[tron_env] Volumetric fog configured.")


# ---------------------------------------------------------------------------
# Ground — emissive blue grid plane
# ---------------------------------------------------------------------------

def _add_grid_plane(stage: Usd.Stage):
    """Create a large flat plane with an emissive blue grid material."""
    plane_path = "/World/TronGrid"

    # Mesh: large flat quad at y=0
    mesh = UsdGeom.Mesh.Define(stage, plane_path)
    size = 50.0
    mesh.CreatePointsAttr([
        Gf.Vec3f(-size, 0.0, -size),
        Gf.Vec3f( size, 0.0, -size),
        Gf.Vec3f( size, 0.0,  size),
        Gf.Vec3f(-size, 0.0,  size),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 1, 0)] * 4)

    # UV coords for tiling the grid texture
    texcoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    scale = 20.0  # grid tile frequency
    texcoords.Set([
        Gf.Vec2f(0, 0), Gf.Vec2f(scale, 0),
        Gf.Vec2f(scale, scale), Gf.Vec2f(0, scale)
    ])

    # Create and bind the grid material
    mat_path = _create_grid_material(stage)
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(mat_path))
    )
    carb.log_info("[tron_env] Tron grid plane created.")


def _create_grid_material(stage: Usd.Stage) -> str:
    """OmniPBR material: dark base + emissive blue grid lines via texture."""
    mat_path = "/World/Looks/TronGridMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = mat_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("OmniPBR")

    # Dark base color
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.01, 0.01, 0.02)
    )
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("roughness_constant", Sdf.ValueTypeNames.Float).Set(0.2)

    # Emissive: blue grid glow
    # No texture file needed — use emissive color + low intensity for ambient glow
    # For full grid lines you'd reference a grid PNG; this gives a solid blue glow
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    # #64d5d2 → linear RGB approx (0.141, 0.816, 0.816) after gamma
    shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.141, 0.816, 0.816)   # Gary Gigabytes teal #64d5d2
    )
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(0.35)

    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return mat_path


# ---------------------------------------------------------------------------
# Lighting — cinematic three-point in blue tones
# ---------------------------------------------------------------------------

def _add_cinematic_lighting(stage: Usd.Stage):
    """
    Three-point lighting rig in blue/cyan tones.
    Key light:  blue-white from above-front
    Fill light: dim cyan from side
    Rim light:  bright cyan from behind (gives drone edge glow)
    """
    lights = [
        {
            "path": "/World/Lights/KeyLight",
            "type": UsdLux.DistantLight,
            "intensity": 800,
            "color": Gf.Vec3f(0.39, 0.84, 0.82),   # #64d5d2 teal-white key
            "rotate": Gf.Vec3f(-45, 30, 0),
        },
        {
            "path": "/World/Lights/FillLight",
            "type": UsdLux.DistantLight,
            "intensity": 200,
            "color": Gf.Vec3f(0.18, 0.30, 0.36),   # #2e4c5b navy fill
            "rotate": Gf.Vec3f(-20, -60, 0),
        },
        {
            "path": "/World/Lights/RimLight",
            "type": UsdLux.DistantLight,
            "intensity": 1400,
            "color": Gf.Vec3f(0.39, 0.84, 0.82),   # strong teal rim — edge glow
            "rotate": Gf.Vec3f(10, 150, 0),
        },
    ]

    for cfg in lights:
        light = cfg["type"].Define(stage, cfg["path"])
        light.CreateIntensityAttr(cfg["intensity"])
        light.CreateColorAttr(cfg["color"])
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.AddRotateXYZOp().Set(cfg["rotate"])

    carb.log_info("[tron_env] Cinematic lighting rig applied.")


# ---------------------------------------------------------------------------
# Drone material — blue emissive LED trim
# ---------------------------------------------------------------------------

def _create_blue_emissive_material(stage: Usd.Stage) -> str:
    """Creates a bright blue emissive OmniPBR material for drones."""
    mat_path = "/World/Looks/TronDroneMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = mat_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("OmniPBR")

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.08, 0.12, 0.12)   # near-black body with teal tint (#141818 family)
    )
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("roughness_constant", Sdf.ValueTypeNames.Float).Set(0.1)
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.141, 0.816, 0.816)   # #64d5d2 — Gary Gigabytes signature teal
    )
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(10.0)

    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    carb.log_info("[tron_env] Blue emissive drone material created.")
    return mat_path


def _apply_material_to_drone(stage: Usd.Stage, drone_prim_path: str, mat_path: str):
    """Recursively apply material to all mesh children of a drone prim."""
    prim = stage.GetPrimAtPath(drone_prim_path)
    if not prim.IsValid():
        carb.log_warn(f"[tron_env] Prim not found: {drone_prim_path}")
        return
    material = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(child).Bind(material)


# ---------------------------------------------------------------------------
# Camera rigs
# ---------------------------------------------------------------------------

class TronCameraRig:
    """
    Manages cinematic camera positions for the Tron showcase.

    Usage:
        cam = TronCameraRig(sim, mode="orbit")
        # In your render loop:
        cam.step()   # advances orbit animation

    Modes:
        "orbit"      — slow 360° orbit around swarm centroid
        "top_down"   — locked overhead view, great for formation shape reveal
        "low_angle"  — dramatic ground-level looking up at swarm
        "chase"      — follows swarm centroid (update centroid each step)
    """

    def __init__(self, sim, mode: str = "orbit",
                 target: Gf.Vec3f = Gf.Vec3f(0, 1.0, 0),
                 radius: float = 8.0,
                 height: float = 2.5,
                 orbit_speed_deg: float = 0.3):
        self.sim = sim
        self.mode = mode
        self.target = target
        self.radius = radius
        self.height = height
        self.orbit_speed = orbit_speed_deg
        self._angle = 0.0

        stage = omni.usd.get_context().get_stage()
        self._cam_path = "/World/TronCamera"
        self._camera = UsdGeom.Camera.Define(stage, self._cam_path)
        self._camera.CreateFocalLengthAttr(24.0)   # wide angle — cinematic
        self._camera.CreateFStopAttr(1.4)           # shallow depth of field
        self._xform = UsdGeom.Xformable(self._camera.GetPrim())
        self._translate_op = self._xform.AddTranslateOp()
        self._rotate_op = self._xform.AddRotateXYZOp()

        self._apply_mode()
        carb.log_info(f"[tron_env] TronCameraRig initialized in '{mode}' mode.")

    def _apply_mode(self):
        if self.mode == "top_down":
            self._translate_op.Set(Gf.Vec3d(self.target[0], 18.0, self.target[2]))
            self._rotate_op.Set(Gf.Vec3d(-90, 0, 0))
        elif self.mode == "low_angle":
            self._translate_op.Set(Gf.Vec3d(self.target[0] - 6.0, 0.3, self.target[2]))
            self._rotate_op.Set(Gf.Vec3d(-8, 20, 0))
        elif self.mode == "orbit":
            self._update_orbit()
        elif self.mode == "chase":
            self._update_orbit()

    def step(self, centroid: Gf.Vec3f | None = None):
        """Call each render step to advance animation."""
        if centroid is not None:
            self.target = centroid
        if self.mode in ("orbit", "chase"):
            self._angle += self.orbit_speed
            if self._angle >= 360.0:
                self._angle -= 360.0
            self._update_orbit()

    def _update_orbit(self):
        rad = math.radians(self._angle)
        x = self.target[0] + self.radius * math.sin(rad)
        z = self.target[2] + self.radius * math.cos(rad)
        y = self.target[1] + self.height
        self._translate_op.Set(Gf.Vec3d(x, y, z))
        yaw = -math.degrees(math.atan2(math.sin(rad), math.cos(rad)))
        pitch = -math.degrees(math.atan2(self.height, self.radius)) * 0.7
        self._rotate_op.Set(Gf.Vec3d(pitch, yaw, 0))

    def set_mode(self, mode: str):
        """Switch camera mode mid-session."""
        self.mode = mode
        self._apply_mode()
        carb.log_info(f"[tron_env] Camera switched to '{mode}' mode.")

    @property
    def prim_path(self) -> str:
        return self._cam_path


# ---------------------------------------------------------------------------
# NVENC video recorder helper
# ---------------------------------------------------------------------------

class TronRecorder:
    """
    Thin wrapper around Isaac Sim's NVENC recorder.

    Usage:
        rec = TronRecorder(output_dir="videos", prefix="showcase")
        rec.start()
        # ... render loop ...
        rec.stop()   # saves videos/showcase_<timestamp>.mp4
    """

    def __init__(self, output_dir: str = "videos", prefix: str = "ggswarm_tron",
                 fps: int = 60, width: int = 1920, height: int = 1080):
        self.output_dir = output_dir
        self.prefix = prefix
        self.fps = fps
        self.width = width
        self.height = height
        self._recorder = None

    def start(self):
        try:
            import omni.kit.capture.viewport as cap
            self._recorder = cap.CaptureExtension()
            self._recorder.start(
                output_folder=self.output_dir,
                file_name_prefix=self.prefix,
                fps=self.fps,
                width=self.width,
                height=self.height,
                capture_every_nth_frame=1,
            )
            carb.log_info(f"[tron_env] Recording started → {self.output_dir}/{self.prefix}_*.mp4")
        except Exception as e:
            carb.log_warn(f"[tron_env] Recorder unavailable: {e}. Use OBS as fallback.")

    def stop(self):
        if self._recorder is not None:
            self._recorder.stop()
            carb.log_info("[tron_env] Recording stopped.")


# ---------------------------------------------------------------------------
# Quick-start: standalone demo (run via isaaclab.bat -p tron_env.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal standalone test — spawns the Tron environment with no drones.
    Just verifies lighting, fog, and grid are working before integrating
    with your play.py.

    Run:
        ..\IsaacLab\isaaclab.bat -p scripts/tron_env.py
    """
    import omni.isaac.kit
    kit = omni.isaac.kit.SimulationApp({"headless": False, "width": 1920, "height": 1080})

    import omni.isaac.core.utils.stage as stage_utils
    stage_utils.create_new_stage()

    from omni.isaac.core import SimulationContext
    sim = SimulationContext(dt=1/60)

    setup_tron_environment(sim)

    cam = TronCameraRig(sim, mode="orbit", radius=10.0, height=3.0)

    sim.reset()
    for i in range(600):   # 10 seconds at 60fps
        cam.step()
        sim.step(render=True)

    kit.close()
