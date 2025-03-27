from dotenv import load_dotenv
from groundlight import ExperimentalApi, BBoxGeometry
from box_orientation.cameras import Cameras
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import uuid
import os
import concurrent.futures

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
        self.view_names = ["top", "front", "side"]

        # the names of the faces of the box
        self.box_faces = ["front", "back", "left", "right", "top", "bottom"]

        self.cameras = Cameras(view_names=self.view_names)

        # setup an object detector for each camera view. It will be used to detect the box in the image
        self.object_detectors = self._initialize_object_detectors()

    def _initialize_object_detectors(self):
        """
        Initializes an object detector for each camera view.
        """

        object_detectors = {}

        for view_name in self.view_names:
            detector = self.gl.create_counting_detector(
                name=f"{view_name}_box_detector_{uuid.uuid4()}",
                query="How many cardboard boxes are in the image?",
                class_name="cardboard_box",
                max_count=1,
                confidence_threshold=0.9,
            )
            object_detectors[view_name] = detector

        return object_detectors

    def onboard_box(self, images_path: str = None, visualize: bool = False):
        """
        Sets up the class to determine the orientation of a box. Either loads images from disk
        or guides the user through capturing and saving images of each face.

        Args:
            images_path (str, optional): Path to directory containing box face images. Images should be named 'front.jpg', 'back.jpg', etc.
            visualize (bool, optional): Whether to visualize the box faces after they are captured or loaded from disk.
        """
        box_face_images = {}

        if images_path:
            # Load images from disk
            for face in self.box_faces:
                image_path = os.path.join(images_path, f"{face}.jpg")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(
                        f"Missing image for {face} face at {image_path}"
                    )
                box_face_images[face] = Image.open(image_path)
        else:
            # Create directory for saving images if it doesn't exist
            save_dir = "box_face_images"
            os.makedirs(save_dir, exist_ok=True)

            # Capture images interactively
            for face in self.box_faces:
                print(
                    f"Please show the {face} face of the box to the top camera and press ENTER to capture"
                )
                input("Press ENTER to capture (or Ctrl+C to quit)")
                image = self.cameras.capture()["top"]
                box_face_images[face] = image

                # Save the captured image
                save_path = os.path.join(save_dir, f"{face}.jpg")
                image.save(save_path)
                print(f"Saved {face} face image to {save_path}")

        self.box_face_images = box_face_images
        if visualize:
            self._visualize_box_faces(self.box_face_images, title="Box Faces")

        self._crop_box_faces(self.box_face_images)

    def _crop_box_faces(self, box_face_images: dict[str, Image.Image]):
        """
        Crops box_face_images to the bounding box of the cardboard box.

        Returns a dictionary of the cropped images.
        """

        submissions = []
        face_names = []  # Keep track of face names in same order as submissions
        for face_name, image in box_face_images.items():
            submissions.append(
                {
                    "detector": self.object_detectors["top"],
                    "image": image,
                    "metadata": {"face_name": face_name},
                }
            )
            face_names.append(face_name)

        results = self._submit_and_wait_for_queries(submissions)

        box_drawn_images = {}
        for face_name, result in zip(face_names, results):
            box_drawn_images[face_name] = self._draw_box(
                box_face_images[face_name], result.rois[0]
            )

        self._visualize_box_faces(
            box_drawn_images, title="Box Faces with Bounding Boxes"
        )

        cropped_images = {}
        # Match results with the correct face names using the stored order
        for face_name, result in zip(face_names, results):
            print(f"{face_name=}, {result.metadata=}")
            cropped_images[face_name] = self._crop_image(
                box_face_images[face_name], result.rois[0]
            )

        self._visualize_box_faces(cropped_images, title="Cropped Box Faces")

        return cropped_images

    def _visualize_box_faces(
        self, box_face_images: dict[str, Image.Image], title: str = None
    ):
        """
        Visualizes the box faces in a 2x3 grid layout using matplotlib.
        Each face of the box will be displayed in its own subplot.
        """
        # Create a figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Plot each face
        for idx, (face_name, image) in enumerate(box_face_images.items()):
            ax = axes[idx]
            ax.imshow(image)
            ax.set_title(face_name)
            ax.axis("off")

        # Turn off any unused subplots
        for idx in range(len(self.box_face_images), 6):
            axes[idx].set_visible(False)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    def _submit_image_queries(self, submissions: list[dict]):
        """
        Submit multiple image queries in parallel using ThreadPoolExecutor.

        Args:
            submissions (list[dict]): A list of dictionaries containing "detector" and "image" keys and optionally "metadata".
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all queries to the thread pool
            futures = [
                executor.submit(
                    self.gl.ask_async,
                    detector=submission["detector"],
                    image=submission["image"],
                    metadata=submission.get("metadata", None),
                )
                for submission in submissions
            ]
            # Wait for all futures to complete and return results
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        return results

    def _submit_and_wait_for_queries(
        self,
        submissions: list[dict],
        timeout_sec: float = 60.0,
    ):
        """
        Submit multiple image queries in parallel and wait for their results.

        Args:
            submissions: List of dictionaries containing "detector" and "image" keys and optionally "metadata".
            timeout_sec: Maximum time to wait for results

        Returns:
            list: List of completed image query results
        """

        def submit_and_wait(submission):
            # Submit the query
            query = self.gl.ask_async(
                detector=submission["detector"],
                image=submission["image"],
                metadata=submission.get("metadata", None),
            )

            # Wait for the result
            return self.gl.wait_for_confident_result(
                image_query=query,
                timeout_sec=timeout_sec,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create futures and store them in order
            futures = []
            for submission in submissions:
                future = executor.submit(submit_and_wait, submission)
                futures.append(future)

            # Wait for all futures to complete in the original order
            results = []
            for future in futures:
                results.append(future.result())

        return results

    def _crop_image(self, image: Image.Image, bbox: BBoxGeometry):
        """
        Crops image to the bounding box.

        Args:
            image (Image.Image): PIL Image to crop
            bbox (BBoxGeometry): Bounding box with normalized coordinates (0-1)

        Returns:
            Image.Image: Cropped image
        """
        # Get image dimensions
        width, height = image.size

        # Convert normalized coordinates to pixel coordinates
        left = int(bbox.geometry.left * width)
        top = int(bbox.geometry.top * height)
        right = int(bbox.geometry.right * width)
        bottom = int(bbox.geometry.bottom * height)

        # Crop and return the image
        return image.crop((left, top, right, bottom))

    def _draw_box(
        self,
        image: Image.Image,
        bbox: BBoxGeometry,
        color: tuple = (255, 0, 0),
        width: int = 3,
    ):
        """
        Draws a bounding box on the image.

        Args:
            image (Image.Image): PIL Image to draw on
            bbox (BBoxGeometry): Bounding box with normalized coordinates (0-1)
            color (tuple): RGB color tuple for the box (default: red)
            width (int): Width of the box lines in pixels (default: 3)

        Returns:
            Image.Image: Image with drawn bounding box
        """
        # Create a copy to avoid modifying the original
        draw_image = image.copy()

        # Get image dimensions
        img_width, img_height = image.size

        # Convert normalized coordinates to pixel coordinates
        left = int(bbox.geometry.left * img_width)
        top = int(bbox.geometry.top * img_height)
        right = int(bbox.geometry.right * img_width)
        bottom = int(bbox.geometry.bottom * img_height)

        # Draw the rectangle
        draw = ImageDraw.Draw(draw_image)
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=width)

        return draw_image


if __name__ == "__main__":
    orientation = BoxOrientation()
    # onboard the box interactively
    # orientation.onboard_box()
    # onboard the box from images on disk
    orientation.onboard_box(images_path="box_face_images")
