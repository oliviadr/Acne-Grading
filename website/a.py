import torchvision.transforms as transforms
from PIL import Image

def pad_image_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width + 1) // 2,
        (max_side - height + 1) // 2
    )
    return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

transform = transforms.Compose([
    transforms.Lambda(pad_image_to_square),
    transforms.Resize(224),
    transforms.ToTensor(),
])

# Example usage
image = Image.open('C:/Users/olivia/OneDrive/Desktop/a/uploads/5.jpg')
transformed_image = transform(image)
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# Example usage
image = Image.open('C:/Users/olivia/OneDrive/Desktop/a/uploads/5.jpg')
transformed_image = transform(image)

# Convert the tensor back to an image
transformed_image_pil = F.to_pil_image(transformed_image)

# Display the image
transformed_image_pil.show()

# Save the transformed image
transformed_image_pil.save('C:/Users/olivia/OneDrive/Desktop/a/uploads/5_transformed.jpg')
