from dotenv import load_dotenv
from groundlight import ExperimentalApi
from box_orientation.cameras import Cameras

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

        # where the cameras are positioned relative to the conveyor belt
        view_names = ["top", "front", "side"]

        self.cameras = Cameras(view_names=view_names)

    def setup(self):
        """
        Sets up the class to to determine the orientation of a box. It guides the user through collecting a sample image of each face of the box using the top camera. It saves the images to self.box_face_images
        """

        box_faces = [
            "front",
            "back",
            "left",
            "right",
            "top",
            "bottom",
        ]

        box_face_images = {}

        for face in box_faces:
            print(
                f"Please show the {face} face of the box to the top camera and press ENTER to capture"
            )

            # Wait for enter press
            input("Press ENTER to capture (or Ctrl+C to quit)")
            # select the relevant image
            image = self.cameras.capture()["top"]
            box_face_images[face] = image

        self.box_face_images = box_face_images


if __name__ == "__main__":
    orientation = BoxOrientation()
    orientation.setup()
