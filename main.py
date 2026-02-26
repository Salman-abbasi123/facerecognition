from dataset import capture_images
from face_recong import start_face_recognition

# Flag to ensure dataset is only loaded once
dataset_loaded = False

# Function to start detection (no reloading of dataset)
def start_detection():
    global dataset_loaded
    if not dataset_loaded:
        print("Please wait while I load the dataset.")
        start_face_recognition('dataset')  # Load dataset for the first time
        dataset_loaded = True
    else:
        print("Starting face detection.")
        start_face_recognition('dataset')  # Start detection without reloading dataset

# Main menu to handle commands
def main_menu():
    print("Welcome! Please select an option:")
    while True:
        print("\n1. Add dataset")
        print("2. Start detection")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            person_name = input("Enter the name of the person for whom the dataset is being added: ").strip()
            if person_name:
                capture_images(person_name)  # Directly use the name in the capture_images function
            else:
                print("Name cannot be empty.")
        elif choice == '2':
            start_detection()  # Start face detection
        elif choice == '3':
            print("Exiting the system.")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the main menu
main_menu()
