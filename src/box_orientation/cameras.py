from framegrab import FrameGrabber
from imgcat import imgcat
import cv2
import matplotlib.pyplot as plt


class Cameras:
    """
    This class is responsible for camera management for the box orientation application.
    """

    def __init__(self):
        # maps view name to a FrameGrabber object
        self.grabbers = {}

    def add_cameras(self, view_names: list[str]):
        """
        Adds a camera view to the class.

        If the camera views are not properly assigned (e.g. the top view is assigned to the side camera, simply update the order of view_names)

        Args:
            serial_number (str): The serial number of the camera.
            view_name (str): The name of the view.
        """

        configs = []
        for view_name in view_names:
            config = {"name": view_name, "input_type": "generic_usb"}
            configs.append(config)

        grabbers = FrameGrabber.create_grabbers(configs=configs)

        for view_name, grabber in grabbers.items():
            self.grabbers[view_name] = grabber

        print(f"Added {len(self.grabbers)} cameras")

    def capture(self):
        """
        Captures a frame from each camera and returns a dictionary mapping view names to images.
        """
        if len(self.grabbers) == 0:
            raise ValueError("No cameras added. Call add_cameras() first.")

        frames = {}
        for view_name, grabber in self.grabbers.items():
            bgr_frame = grabber.grab()
            frames[view_name] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return frames

    def visualize(self, frames: dict, method: str):
        """
        Given the output of capture(), visualize the frames in the CLI or a window.

        Args:
            frames (dict): The output of capture().
            method (str): "cli" or "window"
        """
        if method == "cli":
            for view_name, frame in frames.items():
                print(f"Visualizing {view_name} frame:")
                imgcat(frame)
        elif method == "window":
            # Calculate number of subplots needed
            n_views = len(frames)

            # Create a figure with subplots arranged horizontally
            fig, axes = plt.subplots(1, n_views, figsize=(5 * n_views, 5))

            # Handle case where there's only one view (axes won't be array)
            if n_views == 1:
                axes = [axes]

            # Plot each view
            for (view_name, frame), ax in zip(frames.items(), axes):
                ax.imshow(frame)
                ax.set_title(view_name)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

    def __del__(self):
        """
        Destructor to clean up camera resources when the class is destroyed.
        """
        for grabber in self.grabbers.values():
            grabber.release()


if __name__ == "__main__":
    cameras = Cameras()
    cameras.add_cameras(view_names=["top", "side", "front"])
    frames = cameras.capture()
    cameras.visualize(frames, method="window")
