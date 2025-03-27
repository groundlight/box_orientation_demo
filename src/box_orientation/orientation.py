from dotenv import load_dotenv
from groundlight import ExperimentalApi
from PIL import Image

# loads Groundlight API key from .env file
load_dotenv()


class BoxOrientation:
    """
    This class demonstrates how to use the Groundlight SDK to determine the orientation of a box.

    Physical setup:
    - This class expects a cardboard box on a desk with 3 USB cameras positioned: above, in front, and to the side of the box. See the README for images of the physical setup.

    """

    def __init__(self):
        # the Groundlight client, which will be used to make requests to the Groundlight API
        # We use the ExperimentalApi
        self.gl = ExperimentalApi()

        # a dictionary of views from the cameras
        self.views = None

    def get_views(self) -> dict[str, Image.Image]:
        """
        Obtains a dictionary of views from the cameras and saves them to self.views
        {
            "top": Image.Image,
            "front": Image.Image,
            "side": Image.Image,
        }
        """
        pass

    def setup(self):
        """
        Sets up the class to to determine the orientation of a box. It guides the user through collecting a sample image of each side of the box using the top camera.
        """
