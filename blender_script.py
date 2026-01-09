"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy  # pylint: disable=import-error
import numpy as np
from mathutils import Matrix, Vector  # pylint: disable=no-name-in-module

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
    "dae": bpy.ops.wm.collada_import,
    "fbx": bpy.ops.import_scene.fbx,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "obj": bpy.ops.import_scene.obj,
    "ply": bpy.ops.import_mesh.ply,
    "stl": bpy.ops.import_mesh.stl,
    "usd": bpy.ops.import_scene.usd,
    "usda": bpy.ops.wm.usd_import,
    "usdz": bpy.ops.wm.usd_import,
}

# Dictionary to map file formats to custom suffixes or extensions
FORMAT_EXTENSIONS: Dict[str, str] = {
    "BMP": "bmp",
    "IRIS": "iris",
    "PNG": "png",
    "JPEG": "jpg",
    "JPEG2000": "jp2",
    "TARGA": "tga",
    "TARGA_RAW": "raw",
    "CINEON": "cin",
    "DPX": "dpx",
    "OPEN_EXR_MULTILAYER": "exr",
    "OPEN_EXR": "exr",
    "HDR": "hdr",
    "TIFF": "tiff",
    "AVI_JPEG": "avi",
    "AVI_RAW": "avi",
    "FFMPEG": "mp4",
    "WEBP": "webp",
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera  # pylint: disable=possibly-used-before-assignment


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """
    Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)

    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """
    Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    vec = np.array([0, 0, 0])

    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]

        if maxz > vec[2] > minz:
            correct = True

    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """
    Randomizes the camera location and rotation inside of a spherical shell.
    Move center of MESH in scene to (0,0,0)

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )

    camera = bpy.data.objects["Camera"]

    # Only positive z if required
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector((x, y, z))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    scene = bpy.context.scene

    # Move all objects center point to (0,0,0)
    for obj in scene.objects:
        if obj.type == "MESH":
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # Calculate the local bounding box center
            bbox_center_local = (
                sum((Vector(corner) for corner in obj.bound_box), Vector()) / 8.0
            )
            print(f"BBOX_CENTER (Local): {bbox_center_local}")

            # Check if object already offset to center
            is_already_offset: Vector = obj.location + bbox_center_local
            if is_already_offset != Vector((0, 0, 0)):
                offset: Vector = -bbox_center_local
                print(f"Original object position: {obj.location}")
                obj.location += offset
                bpy.context.view_layer.update()
                print(f"New Object position: {obj.location}")
            else:
                print("Already at center")

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """
    Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")

    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """
    Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy

    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """
    Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """
    Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min: Tuple[float, float, float] = (math.inf,) * 3
    bbox_max: Tuple[float, float, float] = (-math.inf,) * 3
    found: bool = False

    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True

        for coord in obj.bound_box:
            coord = Vector(coord)

            if not ignore_matrix:
                coord = obj.matrix_world @ coord

            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """
    Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """
    Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )

    return RT


def delete_invisible_objects() -> None:
    """
    Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")

    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)

    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """
    Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale: float = 1 / max(bbox_max - bbox_min)

    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset: float = -(bbox_min + bbox_max) / 2

    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """
    Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count: int = 0
    out_files: List[str] = []
    file_path_to_color: Dict[str, List[float]] = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image

                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color: List[float] = [
                                        random.random() for _ in range(3)
                                    ]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs["Base Color"].default_value = (
                                    file_path_to_color[file_path]
                                )

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)

    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """
    Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """
    Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes

    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color

    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """
    Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)

    return rand_color


class MetadataExtractor:
    """
    Class to extract metadata from a Blender scene.

    Args:
        object_path (str): Path to the object file.
        scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
        bdata (bpy.types.BlendData): The current blender data from `bpy.data`.
    """

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count: int = 0

        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)

        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count: int = 0

        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)

        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count: int = 0

        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)

        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths: Set[str] = self._get_image_filepaths()
        material_filepaths: Set[str] = self._get_material_filepaths()
        linked_libraries_filepaths: Set[str] = self._get_linked_libraries_filepaths()

        all_filepaths: Set[str] = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")

        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths: Set[str] = set()

        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))

        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths: Set[str] = set()

        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image

                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))

        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths: Set[str] = set()

        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))

        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count: int = 0

        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys

                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key

        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count: int = 0

        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1

        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "animation_count": self.get_animation_count(),
            "armature_count": self.get_armature_count(),
            "edge_count": self.get_edge_count(),
            "file_size": self.read_file_size(),
            "lamp_count": self.get_lamp_count(),
            "linked_files": self.get_linked_files(),
            "material_count": self.get_material_count(),
            "mesh_count": self.get_mesh_count(),
            # missing_textures
            "object_count": self.get_object_count(),
            "poly_count": self.get_poly_count(),
            # random_color
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "vert_count": self.get_vertex_count(),
        }


def setup_nodes(
    output_path: str,
    capturing_material_alpha: bool = False,
    render_alpha_maps: bool = False,
    render_depth_maps: bool = False,
    render_disparity_maps: bool = False,
) -> None:
    """
    Sets up the Blender compositor nodes to output various image channels, depth maps,
    and disparity maps to specific file paths.

    Args:
        output_path (str): The base path for saving output files, which will be appended
            with suffixes for different channels (e.g., "_r", "_g", "_b", "_a", "_depth", "_disparity").
        capturing_material_alpha (bool): If True, captures the material alpha separately
            instead of standard RGBA channels. Defaults to False.
        render_alpha_maps (bool): If we want to render alpha images of each render (Default: False)
        render_depth_maps (bool): If we want to render depth and disparity images (Default: False)

    Returns:
        None
    """
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants.
    def node_op(op: str, *args, clamp: bool = False) -> bpy.types.NodeSocketFloat:
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op

        if clamp:
            node.use_clamp = True

        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])

        return node.outputs[0]

    def node_clamp(
        value_to_clamp: float, maximum: float = 1.0
    ) -> bpy.types.NodeSocketFloat:
        return node_op("MINIMUM", value_to_clamp, maximum)

    def node_mul(x: Any, y: Any, **kwargs) -> bpy.types.NodeSocketFloat:
        return node_op("MULTIPLY", x, y, **kwargs)

    def node_add(x: float, y: float) -> bpy.types.NodeSocketFloat:
        return node_op("ADD", x, y)

    def node_divide(x: float, y: float) -> bpy.types.NodeSocketFloat:
        return node_op("DIVIDE", x, y)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets: dict[str, bpy.types.NodeSocketFloat] = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]

        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear CIE-XYZ E"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]

    split_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    tree.links.new(color_socket, split_node.inputs[0])

    if render_alpha_maps:
        # Only render the alpha channel
        channel_idx = 3 if not capturing_material_alpha else 0
        channel = "a" if not capturing_material_alpha else "MatAlpha"

        # Create a new output node for the alpha channel
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_{channel}"

        # Link the appropriate output to the output node
        links.new(split_node.outputs[channel_idx], output_node.inputs[0])

        # Exit the function early if capturing_material_alpha is set
        if capturing_material_alpha:
            return

    if render_depth_maps or render_disparity_maps:
        depth_socket = input_sockets["Depth"]
        depth_out = node_clamp(value_to_clamp=node_mul(input_sockets["Depth"], 1 / 16))

        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_depth"
        links.new(depth_out, output_node.inputs[0])
        if render_disparity_maps:
            # Disparity processing
            epsilon: float = 0.0001  # Small constant to avoid division by zero
            depth_with_epsilon = node_add(x=depth_socket, y=epsilon)
            disparity_out = node_divide(x=1.0, y=depth_with_epsilon)

            # Save disparity map
            disparity_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
            disparity_output_node.base_path = f"{output_path}_disparity"
            links.new(disparity_out, disparity_output_node.inputs[0])


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
    fast_mode: bool,
    render_alpha_maps: bool = False,
    render_depth_maps: bool = False,
    render_disparity_maps: bool = False,
) -> None:
    """
    Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.
        render_alpha_maps (bool): If we want to render alpha images of each render (Default: False)
        render_depth_maps (bool): If we want to render depth and disparity images (Default: False)

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # normalize the scene
    normalize_scene()

    # randomize the lighting
    randomize_lighting()

    # render the images
    for i in range(num_renders):
        # set camera
        camera = randomize_camera(
            only_northern_hemisphere=only_northern_hemisphere,
        )

        # Get the current file format for image output
        current_format: str = render.image_settings.file_format

        # Check if the current file format is in our dictionary
        if current_format in FORMAT_EXTENSIONS:
            # Get the suffix for the current file format
            ext = FORMAT_EXTENSIONS[current_format]
        else:
            # Default suffix if format is not recognized
            ext = "png"

        output_path = os.path.join(output_dir, f"{i:03d}.{ext}")

        use_workbench = bpy.context.scene.render.engine == "BLENDER_WORKBENCH"
        if use_workbench:
            # We must use a different engine to compute depth maps.
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
            bpy.context.scene.eevee.taa_render_samples = (
                1  # faster, since we discard image.
            )

        if bpy.context.scene.render.engine == "CYCLES":
            # We should still impose a per-frame time limit
            # so that we don't timeout completely.
            bpy.context.scene.cycles.time_limit = 40

        # render the image
        bpy.context.view_layer.update()
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.filepath = output_path
        if render_depth_maps or render_disparity_maps or render_alpha_maps:
            setup_nodes(
                output_path=output_path,
                render_alpha_maps=render_alpha_maps,
                render_depth_maps=render_depth_maps,
                render_disparity_maps=render_disparity_maps,
            )
            bpy.ops.render.render(write_still=True)

            # The output images must be moved from their own sub-directories, or
            # discarded if we are using workbench for the color.

            # render disparity maps and maybe depth maps too
            if render_disparity_maps:
                for channel_name in ["depth", "disparity"]:
                    sub_dir = f"{output_path}_{channel_name}"
                    image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])

                    name, ext = os.path.splitext(output_path)
                    if channel_name == "disparity":
                        os.rename(image_path, f"{name}_{channel_name}{ext}")
                    else:
                        if render_depth_maps:
                            os.rename(image_path, f"{name}_{channel_name}{ext}")
                        else:
                            os.remove(image_path)

                    os.removedirs(sub_dir)
            # only render depth maps and not disparity maps
            if render_depth_maps and not render_disparity_maps:
                for channel_name in ["depth"]:
                    sub_dir = f"{output_path}_{channel_name}"
                    image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])

                    name, ext = os.path.splitext(output_path)
                    if channel_name == "depth" or not use_workbench:
                        os.rename(image_path, f"{name}_{channel_name}{ext}")
                    else:
                        os.remove(image_path)

                    os.removedirs(sub_dir)
            # render alpha maps
            if render_alpha_maps:
                for channel_name in ["a"]:
                    sub_dir = f"{output_path}_{channel_name}"
                    image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])

                    name, ext = os.path.splitext(output_path)
                    if channel_name == "depth" or not use_workbench:
                        os.rename(image_path, f"{name}_{channel_name}{ext}")
                    else:
                        os.remove(image_path)

                    os.removedirs(sub_dir)
        else:
            render_path = os.path.join(output_dir, f"{i:03d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

        if use_workbench:
            # Re-render RGBA using workbench with texture mode, since this seems
            # to show the most reasonable colors when lighting is broken.
            bpy.context.scene.use_nodes = False
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
            bpy.context.scene.render.image_settings.color_mode = "RGBA"
            bpy.context.scene.render.image_settings.color_depth = "8"
            bpy.context.scene.display.shading.color_type = "TEXTURE"
            bpy.context.scene.display.shading.light = "FLAT"

            if fast_mode:
                # Single pass anti-aliasing.
                bpy.context.scene.display.render_aa = "FXAA"

            os.remove(output_path)

            bpy.ops.render.render(write_still=True)
            bpy.context.scene.render.image_settings.color_mode = "RGBA"
            bpy.context.scene.render.image_settings.color_depth = "16"

        # save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(camera)
        rt_matrix_path = os.path.join(output_dir, f"{i:03d}.npy")
        np.save(rt_matrix_path, rt_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--depth_maps",
        action="store_true",  # This action will set the argument to True when provided.
        help="Render depth maps if this flag is set (default: False).",
    )
    parser.add_argument(
        "--disparity_maps",
        action="store_true",  # This action will set the argument to True when provided.
        help="Render disparity maps if this flag is set (default: False).",
    )
    parser.add_argument(
        "--render_alpha",
        action="store_true",  # This action will set the argument to True when provided.
        help="Render alpha maps if this flag is set (default: False).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        choices=["CYCLES", "BLENDER_EEVEE"],
        help="Render engine to use (CYCLES or BLENDER_EEVEE).",
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        default=False,
        help="Only render the northern hemisphere of the object.",
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )

    parser.add_argument(
        "--render_file_format",
        type=str,
        default="PNG",
        choices=[
            "BMP",
            "IRIS",
            "PNG",
            "JPEG",
            "JPEG2000",
            "TARGA",
            "TARGA_RAW",
            "CINEON",
            "DPX",
            "OPEN_EXR_MULTILAYER",
            "OPEN_EXR",
            "HDR",
            "TIFF",
            "AVI_JPEG",
            "AVI_RAW",
            "FRAMESERVER",
            "H264",
            "FFMPEG",
            "THEORA",
            "XVID",
        ],
        help="Rendered image file format (default is 'PNG').",
    )
    parser.add_argument(
        "--render_colour_mode",
        type=str,
        default="RGBA",
        choices=[
            "BW",
            "RGB",
            "RGBA",
        ],
        help="Rendered image colour mode (default is 'RGBA').",
    )
    parser.add_argument(
        "--resolution_x",
        type=int,
        default=512,
        help="Horizontal resolution of the rendered images.",
    )
    parser.add_argument(
        "--resolution_y",
        type=int,
        default=512,
        help="Vertical resolution of the rendered images.",
    )
    parser.add_argument(
        "--resolution_percentage",
        type=int,
        default=100,
        help="Resolution percentage scale of the rendered images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU"],
        help="Device to use for rendering with Cycles (CPU or GPU).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples for Cycles rendering.",
    )
    parser.add_argument(
        "--diffuse_bounces",
        type=int,
        default=1,
        help="Number of diffuse bounces for Cycles rendering.",
    )
    parser.add_argument(
        "--glossy_bounces",
        type=int,
        default=1,
        help="Number of glossy bounces for Cycles rendering.",
    )
    parser.add_argument(
        "--transparent_max_bounces",
        type=int,
        default=3,
        help="Maximum number of transparent bounces for Cycles rendering.",
    )
    parser.add_argument(
        "--transmission_bounces",
        type=int,
        default=3,
        help="Number of transmission bounces for Cycles rendering.",
    )
    parser.add_argument(
        "--filter_width",
        type=float,
        default=0.01,
        help="Filter width for Cycles rendering.",
    )
    parser.add_argument(
        "--use_denoising",
        action="store_true",
        help="Use denoising for Cycles rendering.",
        default=True,
    )
    parser.add_argument(
        "--film_transparent",
        action="store_true",
        help="Render with transparent film.",
        default=True,
    )
    parser.add_argument(
        "--compute_device_type",
        type=str,
        default="NONE",
        choices=["CUDA", "OPENCL", "OPTIX", "NONE"],
        help="Compute device type for Cycles (CUDA, OPENCL, OPTIX or NONE).",
    )
    parser.add_argument("--fast_mode", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render
    render_depth_maps = args.depth_maps
    render_disparity_maps = args.disparity_maps
    render_alpha_maps = args.render_alpha

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = args.render_file_format  #  "PNG"
    render.image_settings.color_mode = args.render_colour_mode  #  "RGBA"
    render.resolution_x = args.resolution_x  #  512
    render.resolution_y = args.resolution_y  #  512
    render.resolution_percentage = args.resolution_percentage  #  100

    # Set cycles settings
    scene.cycles.device = args.device  # "GPU"
    scene.cycles.samples = args.samples  # 128
    scene.cycles.diffuse_bounces = args.diffuse_bounces  # 1
    scene.cycles.glossy_bounces = args.glossy_bounces  # 1
    scene.cycles.transparent_max_bounces = args.transparent_max_bounces  # 3
    scene.cycles.transmission_bounces = args.transmission_bounces  # 3
    scene.cycles.filter_width = args.filter_width  # 0.01
    scene.cycles.use_denoising = args.use_denoising  # True
    scene.render.film_transparent = args.film_transparent  # True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
        args.compute_device_type
    )  # "CUDA" or "OPENCL"

    path_to_object: str = args.object_path
    object_Path: Path = Path(path_to_object)
    assert object_Path.exists(), f"Invalid object path received: {object_Path}"

    num_renders: int = args.num_renders
    assert num_renders > 0, f"Invalid number of renders received: {num_renders}"

    # Render the images
    render_object(
        object_file=path_to_object,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
        fast_mode=args.fast_mode,
        render_alpha_maps=render_alpha_maps,
        render_depth_maps=render_depth_maps,
        render_disparity_maps=render_disparity_maps,
    )
