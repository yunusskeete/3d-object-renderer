import os
import shutil
import subprocess
from logging import getLogger
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Response
from python_utils.configs.rendering_config import BPYConfig, get_bpy_config
from python_utils.errors import log_and_raise
from python_utils.errors.request_errors import BadRequestError
from python_utils.model_schemas.files import FileUpload, UploadFile
from python_utils.model_schemas.images import ImageFile, ImageFiles
from python_utils.model_schemas.rendering import RenderAssetRequest, RenderAssetResponse
from python_utils.utils.file_utils import read_files_from_folder

load_dotenv(dotenv_path=Path(".env"))

logger = getLogger(__name__)

app = FastAPI()

# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation


@app.get("/ping")
async def ping():
    return "pong"


def run_blender_render_subprocess(
    render_asset_request: RenderAssetRequest,
    object_path: Path,
    output_dir_path: Path,
    bpy_config: Optional[BPYConfig] = None,
):
    bpy_config = bpy_config or get_bpy_config()

    # Build command to run Blender with your script and custom args
    command = [
        "python",
        "blender_script.py",
        "--",
        "--object_path",
        str(object_path),
        "--output_dir",
        str(output_dir_path),
    ]

    bpy_config.num_renders = (
        render_asset_request.rendering_details.num_renders or bpy_config.num_renders
    )
    bpy_config.only_northern_hemisphere = (
        render_asset_request.rendering_details.only_northern_hemisphere
        or bpy_config.only_northern_hemisphere
    )
    bpy_config.fast_mode = (
        render_asset_request.rendering_details.fast_mode or bpy_config.fast_mode
    )
    bpy_config.render_alpha = (
        render_asset_request.rendering_details.render_alpha_maps
        or bpy_config.render_alpha
    )
    bpy_config.depth_maps = (
        render_asset_request.rendering_details.render_depth_maps
        or bpy_config.depth_maps
    )
    bpy_config.disparity_maps = (
        render_asset_request.rendering_details.render_disparity_maps
        or bpy_config.disparity_maps
    )

    # Append each arg
    for arg, value in bpy_config.model_dump(mode="json").items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{arg}")
        elif value:
            command.extend([f"--{arg}", str(value)])

    subprocess.run(command, check=True)


@app.post("/render/asset")
async def render_asset(render_asset_request: RenderAssetRequest) -> RenderAssetResponse:
    uid: str = render_asset_request.uid
    logger.info(f"Handle render request process started: ({uid=})")

    ### Save the file content to a temporary directory
    # Create temp dir
    temp_asset_dir_path: Path = (
        Path(os.environ.get("TEMP_ASSET_FOLDER", "./tmp/render")).resolve() / uid
    )
    temp_asset_dir_path.mkdir(parents=True, exist_ok=True)

    temp_asset_usda_file_path: Optional[Path] = None

    try:
        idx: int
        upload_file: UploadFile
        for idx, upload_file in enumerate(
            render_asset_request.asset_files.upload_files
        ):

            temp_asset_file_path: Path = (
                temp_asset_dir_path / upload_file.file_upload.file_details.filename
            )
            if idx == 0 and temp_asset_file_path.suffix == ".usda":
                temp_asset_usda_file_path = temp_asset_file_path

            temp_asset_file_path.parent.mkdir(exist_ok=True, parents=True)

            with temp_asset_file_path.open("wb") as temp_file:
                temp_file.write(
                    upload_file.file_upload.file_content.get_content_bytes()
                )

        temp_output_path: Path = temp_asset_dir_path / "output"

        if temp_asset_usda_file_path is None:
            return log_and_raise(
                error=BadRequestError(
                    details=".usda asset file path not in asset rendering request"
                )
            )

        # TODO: Don't save and read to disk. Make this return metadata and images

        run_blender_render_subprocess(
            render_asset_request=render_asset_request,
            object_path=temp_asset_usda_file_path,
            output_dir_path=temp_output_path,
        )

        image_file_uploads: List[FileUpload] = read_files_from_folder(
            folder_path=temp_output_path,
            file_format=os.environ.get("RENDER_FILE_FORMAT", "png"),
            acceptable_file_extensions={
                host.strip()
                for host in os.environ.get(
                    "ACCEPTABLE_FILE_EXTENSIONS", "png,jpg,jpeg"
                ).split(",")
                if host.strip()
            },
        )
        image_files: List[ImageFile] = [
            ImageFile(
                file_upload=file_upload,
                uid=uid,
            )
            for file_upload in image_file_uploads
        ]

        return RenderAssetResponse(
            uid=uid,
            rendering_details=render_asset_request.rendering_details,
            image_files=ImageFiles(images=image_files),
        )

    except Exception as e:
        logger.exception(f"Failed to render file {uid}: {e}")
        # TODO: Determine status code
        return Response(content=f"Failed to render file {uid}: {e}", status_code=500)

    finally:
        # Clean up the temporary file
        if temp_asset_dir_path.exists():
            # Verify that files are created
            logger.info(
                f"Directory contents before deletion: {list(temp_asset_dir_path.iterdir())}"
            )

            # Delete the directory and its contents
            if temp_asset_dir_path.exists() and temp_asset_dir_path.is_dir():
                shutil.rmtree(temp_asset_dir_path)
                logger.info(f"Deleted {temp_asset_dir_path}")
            else:
                logger.info(
                    f"{temp_asset_dir_path} does not exist or is not a directory"
                )

            # Verify deletion
            if not temp_asset_dir_path.exists():
                logger.info(f"{temp_asset_dir_path} has been successfully deleted.")
            else:
                logger.warning(f"{temp_asset_dir_path} still exists.")
