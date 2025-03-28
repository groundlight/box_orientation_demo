from dotenv import load_dotenv
from groundlight import ExperimentalApi, BBoxGeometry
from box_orientation.cameras import Cameras
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import uuid
import os
import concurrent.futures
import time

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

        # Replace multiple object detectors with a single one
        self.object_detector = self.gl.create_counting_detector(
            name=f"box_detector_{uuid.uuid4()}",
            query="How many cardboard boxes are in the image?",
            class_name="cardboard_box",
            max_count=1,
            confidence_threshold=0.9,
        )

        # setup a multiclass detector for each camera view. It will be used to determine the box face that is facing each camera.
        self.multiclass_detectors = self._initialize_multiclass_detectors()

    def _initialize_multiclass_detectors(self):
        """
        Initializes a multiclass detector for each camera view.
        """

        multiclass_detectors = {}

        for view_name in self.view_names:
            detector = self.gl.create_multiclass_detector(
                name=f"{view_name}_multiclass_detector_{uuid.uuid4()}",
                query="Which face of the box is facing the camera? See notes and example images for reference.",
                class_names=self.box_faces,
                confidence_threshold=0.9,
            )
            multiclass_detectors[view_name] = detector

        return multiclass_detectors

    def onboard_box(self, images_path: str = None, visualize: bool = False):
        """
        Sets up the class to determine the orientation of a box. Either loads images from disk
        or guides the user through capturing and saving images of each face.

        Then:
        1. Determines the bounding box of the box in each image
        2. Crops each image to just the box
        3. Makes each a square cropped image

        Args:
            images_path (str, optional): Path to directory containing box face images. Images should be named 'front.jpg', 'back.jpg', etc.
            visualize (bool, optional): Whether to visualize the intermediate results.
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

        # crop each image to just the face of the box
        self.cropped_box_face_images = self._crop_box_faces(self.box_face_images)

        # make each cropped image square
        self.square_box_face_images = {
            face_name: self._make_square_image(image)
            for face_name, image in self.cropped_box_face_images.items()
        }

        # Create the matrix image. This will be used as a reference for labelers to determine the orientation of the box.
        self.matrix_image = self._create_image_matrix(self.square_box_face_images)

        if visualize:
            # Display the matrix
            plt.figure(figsize=(10, 15))
            plt.imshow(self.matrix_image)
            plt.axis("off")
            plt.title("Box Face Matrix")
            plt.show()

        # Add a note to each multiclass detector with the image matrix
        self._add_image_matrix_note_to_multiclass_detectors()

        # Add GT labels to each multiclass detector
        self._add_groundtruth_to_multiclass_detectors()

    def _crop_box_faces(
        self, box_face_images: dict[str, Image.Image], visualize: bool = False
    ):
        """
        Crops box_face_images to the bounding box of the cardboard box.

        Args:
            box_face_images (dict[str, Image.Image]): A dictionary of the box face images.
            visualize (bool, optional): Whether to visualize the intermediate results.

        Returns a dictionary of the cropped images.
        """

        submissions = []
        face_names = []  # Keep track of face names in same order as submissions
        for face_name, image in box_face_images.items():
            submissions.append(
                {
                    "detector": self.object_detector,
                    "image": image,
                    "metadata": {"face_name": face_name},
                }
            )
            face_names.append(face_name)

        results = self._submit_and_wait_for_queries(
            submissions=submissions, timeout_sec=20
        )

        box_drawn_images = {}
        for face_name, result in zip(face_names, results):
            box_drawn_images[face_name] = self._draw_box(
                box_face_images[face_name], result.rois[0]
            )

        if visualize:
            self._visualize_box_faces(
                box_drawn_images, title="Box Faces with Bounding Boxes"
            )

        cropped_images = {}
        # Match results with the correct face names using the stored order
        for face_name, result in zip(face_names, results):
            cropped_images[face_name] = self._crop_image(
                box_face_images[face_name], result.rois[0]
            )

        if visualize:
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

    def _make_square_image(
        self, image: Image.Image, target_size: int = 500
    ) -> Image.Image:
        """
        Converts an image to a square with black padding while maintaining aspect ratio.

        Args:
            image (Image.Image): Input PIL Image
            target_size (int): Width and height of the output square image

        Returns:
            Image.Image: Square image with black padding if necessary
        """
        # Create a black background image
        square_img = Image.new("RGB", (target_size, target_size), color="black")

        # Get original image size
        width, height = image.size

        # Calculate the resize ratio to fit within target_size while maintaining aspect ratio
        ratio = min(target_size / width, target_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize the image
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate position to paste (center the image)
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2

        # Paste the resized image onto the black background
        square_img.paste(resized_img, (paste_x, paste_y))

        return square_img

    def _create_image_matrix(
        self, square_images: dict[str, Image.Image], grid_width: int = 4
    ) -> Image.Image:
        """
        Creates a 2x3 matrix of square images with red grid lines between them and face labels.

        Args:
            square_images (dict[str, Image.Image]): Dictionary of square images for each face
            grid_width (int): Width of the grid lines in pixels

        Returns:
            Image.Image: Combined image matrix with grid lines and labels
        """
        # All images should be the same size since they're square
        single_size = list(square_images.values())[0].size[0]

        # Create a new image with size 2x3 times the single image size plus space for grid lines
        matrix_width = single_size * 2 + grid_width
        matrix_height = single_size * 3 + grid_width * 2
        matrix_image = Image.new("RGB", (matrix_width, matrix_height), color="red")

        # Define the order of faces in the matrix (2x3)
        matrix_order = [["front", "back"], ["left", "right"], ["top", "bottom"]]

        draw = ImageDraw.Draw(matrix_image)

        # Use a larger font size and thicker stroke
        font = ImageFont.load_default(size=60)

        # Paste each image in its position, accounting for grid line spacing
        for row_idx, row in enumerate(matrix_order):
            for col_idx, face in enumerate(row):
                # Calculate position with grid line offsets
                x = col_idx * single_size + (col_idx * grid_width)
                y = row_idx * single_size + (row_idx * grid_width)

                # Paste the image
                matrix_image.paste(square_images[face], (x, y))

                # Add text label
                text_x = x + 20
                text_y = y + 20
                draw.text(
                    (text_x, text_y),
                    face.upper(),
                    fill="red",
                    font=font,
                    stroke_width=5,
                    stroke_fill="white",
                )

        return matrix_image

    def _add_image_matrix_note_to_multiclass_detectors(self):
        """
        After we've created the multiclass detectors (during init) and the image matrix (during onboarding),  we need to attach a note to each of the multiclass detectors with this image.
        """
        for detector in self.multiclass_detectors.values():
            self.gl.create_note(
                detector=detector,
                note="See attached image for what each face of the box looks like.",
                image=self.matrix_image,
            )

    def _add_groundtruth_to_multiclass_detectors(self):
        """
        After we've created the multiclass detectors (during init) we can attach a GT labels with images of each face.
        """
        for detector in self.multiclass_detectors.values():
            for face_name, image in self.square_box_face_images.items():
                iq = self.gl.ask_async(
                    detector=detector,
                    image=image,
                    metadata={"face_name": face_name},
                )
                self.gl.add_label(image_query=iq, label=face_name)

    def get_box_orientation(self, visualize: bool = False):
        """
        This function determines the current box orientation

        Args:
            visualize (bool, optional): Whether to stop to visualize intermediate results.
        """

        # 1. Capture an image from each camera view
        # 2. Crop each image to just the box
        # 3. Submit cropped images to relevant GL multiclass detectors
        # 4. Get the predicted class for each image
        # 5. Determine the box orientation based on these predictions

        camera_views = self.cameras.capture()

        # submit each image to the relevant object detector to get the bounding box
        submissions = []
        for view_name, image in camera_views.items():
            submissions.append(
                {
                    "detector": self.object_detector,
                    "image": image,
                    "metadata": {"view_name": view_name},
                }
            )

        results = self._submit_and_wait_for_queries(
            submissions=submissions, timeout_sec=5
        )

        camera_views_with_bounding_boxes = {}
        for view_name, result in zip(camera_views.keys(), results):
            camera_views_with_bounding_boxes[view_name] = self._draw_box(
                camera_views[view_name], result.rois[0]
            )

        if visualize:
            self._visualize_box_faces(
                camera_views_with_bounding_boxes,
                title="Box submission: Camera Views with Bounding Boxes",
            )

        cropped_camera_views = {}
        for result in results:
            view_name = result.metadata["view_name"]
            cropped_camera_views[view_name] = self._crop_image(
                camera_views[view_name], result.rois[0]
            )

        if visualize:
            self._visualize_box_faces(
                cropped_camera_views,
                title="Box submission: Cropped Camera Views",
            )

        # submit each cropped image to the relevant multiclass detector
        submissions = []
        for view_name, image in cropped_camera_views.items():
            submissions.append(
                {
                    "detector": self.multiclass_detectors[view_name],
                    "image": image,
                    "metadata": {"view_name": view_name},
                }
            )

        results = self._submit_and_wait_for_queries(
            submissions=submissions, timeout_sec=5
        )
        # get the label for each view
        labels = {}
        for iq in results:
            labels[iq.metadata["view_name"]] = {
                "label": iq.result.label,
                "confidence": iq.result.confidence,
            }

        # print the results
        for view_name, label in labels.items():
            print(
                f"{view_name} view: {label['label']} face (confidence: {label['confidence']})"
            )
        print()


if __name__ == "__main__":
    orientation = BoxOrientation()
    # onboard the box interactively
    # orientation.onboard_box()

    # onboard the box from images on disk
    orientation.onboard_box(images_path="box_face_images", visualize=True)

    print("Box onboarded")

    while True:
        print("Getting box orientation...")
        orientation.get_box_orientation(visualize=True)
        time.sleep(5)
