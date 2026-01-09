import os
from logging import Logger, getLogger
from typing import Optional


async def run_tests(logger: Optional[Logger] = None) -> None:
    logger = logger or getLogger(__name__)

    # pylint: disable=import-outside-toplevel
    from typing import Callable, List

    from python_utils.model_schemas.common import Base64StringContent
    from python_utils.model_schemas.files import (
        FileDetails,
        FileUpload,
        UploadFile,
        UploadFiles,
    )
    from python_utils.model_schemas.rendering import (
        RenderAssetRequest,
        RenderAssetResponse,
        RenderingDetails,
    )
    from python_utils.utils.file_utils import (
        encode_file_to_base64,
        get_usda_file_path,
        rglob_excluding,
    )

    from app import render_asset

    print("Running startup tests...")

    asset_folder_path: Path = Path(
        "sample_objects/Accessories/Decor/SM_Decor01"
    ).resolve()
    assert (
        asset_folder_path.exists()
    ), f"No asset folder found at path: ({asset_folder_path=})"
    assert (
        asset_folder_path.is_dir()
    ), f"Asset folder path is not a directory: ({asset_folder_path=})"

    asset_usda_file_path: Path = [
        path for path in asset_folder_path.rglob("*.usda")
    ].pop()
    print(f"{asset_usda_file_path=}")
    assert asset_usda_file_path.exists(), f"USDA file not found: {asset_usda_file_path}"

    pattern: str = "*"
    renderings_pattern: str = "_Capture-"
    exclude_condition: Callable[[Path], bool] = lambda p: renderings_pattern in p.name
    include_directories: bool = False
    ignored_ancestors: str = str(Path("sample_objects/").resolve())
    asset_file_paths: List[Path] = rglob_excluding(
        path=asset_folder_path,
        pattern=pattern,
        exclude_condition_met=exclude_condition,
        include_directories=include_directories,
        # ignored_ancestors=ignored_ancestors,
    )
    # Move the usda_file_path to the front of the list
    usda_file_path: Path = get_usda_file_path(asset_file_paths)
    print(f"{usda_file_path=}")
    asset_file_paths.remove(usda_file_path)
    asset_file_paths.insert(0, usda_file_path)
    print(f"{len(asset_file_paths)=}")
    print(f"{asset_file_paths=}")

    asset_rendering_request: RenderAssetRequest = RenderAssetRequest(
        uid="abcde_12345",
        rendering_details=RenderingDetails(
            num_renders=int(os.environ.get("NUM_RENDERS", "6")),
            only_northern_hemisphere=(
                os.environ.get("ONLY_NORTHERN_HEMISPHERE", "true").lower() == "true"
            ),
            fast_mode=(os.environ.get("FAST_MODE", "false").lower() == "true"),
            render_alpha_maps=(
                os.environ.get("RENDER_ALPHA_MAPS", "false").lower() == "true"
            ),
            render_depth_maps=(
                os.environ.get("RENDER_DEPTH_MAPS", "false").lower() == "true"
            ),
            render_disparity_maps=(
                os.environ.get("RENDER_DISPARITY_MAPS", "false").lower() == "true"
            ),
        ),
        asset_files=UploadFiles(
            upload_files=[
                UploadFile(
                    url="",
                    file_upload=FileUpload(
                        file_details=FileDetails(
                            file_stem=file_path.stem, mime_type=file_path.suffix
                        ),
                        file_content=Base64StringContent(
                            base64_string_content=encode_file_to_base64(
                                file_path=file_path
                            )
                        ),
                    ),
                )
                for file_path in asset_file_paths
            ]
        ),
    )

    images_response: RenderAssetResponse = await render_asset(
        render_asset_request=asset_rendering_request
    )

    print(f"{images_response=}")


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn
    from dotenv import load_dotenv

    assert load_dotenv(dotenv_path=Path(".env"))

    if os.environ.get("RUN_STARTUP_TESTS", "false").lower() == "true":
        import asyncio

        asyncio.run(run_tests())

    uvicorn.run(
        "app:app",
        host=os.environ.get("RENDERING_HOST", "localhost"),
        port=int(os.environ.get("RENDERING_PORT", 8001)),
        reload=(os.environ.get("RELOAD_APP", "true").lower() == "true"),
    )
