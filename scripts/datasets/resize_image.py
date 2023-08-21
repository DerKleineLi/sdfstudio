import os

from PIL import Image


def resize_images(source_folder, target_folder, new_size):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(source_folder, filename)
            img = Image.open(image_path)
            img = img.resize(new_size)

            target_path = os.path.join(target_folder, filename)
            img.save(target_path)
            print(f"Resized and saved: {target_path}")

if __name__ == "__main__":
    source_folder = "/cluster/angmar/hli/data/TNT/Meetingroom/images"
    target_folder = "/cluster/angmar/hli/data/TNT/Meetingroom_correct_size/images"
    new_size = (1500, 835)  # Specify the new width and height

    resize_images(source_folder, target_folder, new_size)