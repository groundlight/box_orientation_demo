Todo list for rapid prototyping for box orientation demo. The list is broken down into hardware, software, and deliverable preparation sections.
Within each section, work is organized in order of completion.

# Physical Setup

Owners: Sunil and Harry

- [x] Sunil: Carve out some room to build demo
- [x] Sunil: Construct a cube that we will attempt to predict the position of
- [x] Sunil: Set up conveyer belt
- [x] Harry: Set up at least 2 but preferably 3 USB cameras:
  - Above belt
  - Side of belt
  - In front of belt (stretch goal if we have enough cameras)
- [] Sunil: Add description/photos of physical setup to README

# Software

Owners: Sunil and Harry

Overall design:
In a while loop, we will grab images, localize, crop, and then predict the orientation of the cube.

- [x] Harry: Method that grabs an image from each camera and returns it as a dict. It should have the schema `{<view_name>: <PIL Image>}`.
- [] Sunil: Method of onboarding a new object.
- [] Sunil: Localize the cube in each view with object detection model (bounding box per view).
- [] Sunil: Crop each view to the bounding box.
- [] Sunil: Classify each view with multiclass model.
- [] Sunil: Determine the set of possible orientations of the cube based on the inference results. This might need to be a bit flexible. If no valid orientation is found using all views, we could try seaching for the orientation using only a subset of views.

# Deliverable preparation

## Questions for the customer

- Can you tell me more about the application/problem you are trying to solve?
- What types of boxes are you interested in supporting? Do the sides have different features or notable characteristics?
- Are all orientations (all 24) possible?
- How visually different are the boxes you want to support? Are they all about the same, or do they have very significant differences?
- Is the box always square to the conveyor belt? Or can it be rotated/off-axis?
- How fast does the solution need to be?
- Tell me more about onboarding new objects. Who is going to be handling this?
