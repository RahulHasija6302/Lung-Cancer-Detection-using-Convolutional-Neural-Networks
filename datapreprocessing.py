
//Data Preprocessing


from sklearn.model_selection import train_test_split

# Resize the images to a consistent size suitable for the CNN model
image_size = (128, 128)
images_resized = []
for image in images:
    image_resized = cv2.resize(image, image_size)
    images_resized.append(image_resized)

# Normalize the pixel values of the images to a range between 0 and 1
images_normalized = np.array(images_resized) / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_normalized, labels, test_size=0.2, random_state=42)
