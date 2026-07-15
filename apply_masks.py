import os
import cv2
import numpy as np

dataset_path = r"/home_domuser2/s49imoni/4D_reconstruction/3DGUT/src_code/3dgrut/data/t1_cam1"

image_root = os.path.join(dataset_path, "images")
mask_root = os.path.join(dataset_path, "masks")

extensions = (".png", ".jpg", ".jpeg")


def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h)

    if scale < 1:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return img


# Create resizable windows
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Masked", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Original", 800, 600)
cv2.resizeWindow("Mask", 800, 600)
cv2.resizeWindow("Masked", 800, 600)


# Collect all images first
image_files = []

for root, _, files in os.walk(image_root):
    for file in files:
        if file.lower().endswith(extensions):
            image_files.append(
                os.path.join(root, file)
            )


# Sort images
image_files.sort()


print(f"Found {len(image_files)} images")


for image_path in image_files:

    rel_path = os.path.relpath(image_path, image_root)

    # remove extension
    filename = os.path.splitext(rel_path)[0]

    # create mask path
    mask_path = os.path.join(
        mask_root,
        filename + "_mask.png"
    )


    if not os.path.exists(mask_path):
        print(f"Mask missing: {mask_path}")
        continue


    image = cv2.imread(image_path)

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if image is None or mask is None:
        print("Could not load:", image_path)
        continue


    # Resize mask if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )


    # Convert mask to binary
    mask_binary = (mask > 127).astype(np.uint8)


    # Apply mask
    masked = image * mask_binary[:, :, None]


    # Display
    cv2.imshow(
        "Original",
        resize_for_display(image)
    )

    cv2.imshow(
        "Mask",
        resize_for_display(mask_binary * 255)
    )

    cv2.imshow(
        "Masked",
        resize_for_display(masked)
    )


    print("\nShowing:", image_path)
    print("Press any key -> next image")
    print("Press ESC -> exit")


    key = cv2.waitKey(0) & 0xFF


    if key == 27:   # ESC
        break


cv2.destroyAllWindows()





### to duplicate global mask for camera0 ###

# import os
# import cv2

# # Path to your single mask
# mask_path = r"/home_domuser2/s49imoni/4D_reconstruction/3DGUT/src_code/3dgrut/data/frames/mask/mask.png"

# # Output folder
# output_folder = r"/home_domuser2/s49imoni/4D_reconstruction/kitchen_v1/global_masks"

# # Number of copies
# num_masks = 90

# # Create output folder
# os.makedirs(output_folder, exist_ok=True)

# # Read mask
# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# if mask is None:
#     raise FileNotFoundError(f"Could not load mask: {mask_path}")

# # Save copies
# for i in range(1, num_masks + 1):

#     filename = f"{i:03d}_mask.png"   # gives 001,002,...099

#     save_path = os.path.join(
#         output_folder,
#         filename
#     )

#     cv2.imwrite(save_path, mask)

#     print(f"Saved {save_path}")

# print("Done!")