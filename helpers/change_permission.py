import os
import stat
import argparse

def change_permissions_if_read_only(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                # Get current permissions
                current_permissions = os.stat(file_path).st_mode
                
                # Check if the file is read-only for the owner
                if not (current_permissions & stat.S_IWUSR):
                    # Change permissions to read and write for the owner
                    os.chmod(file_path, current_permissions | stat.S_IWUSR)
                    print(f"Permissions updated for file: {file_path}")
                else:
                    pass
            except Exception as e:
                print(f"Failed to change permissions for file {file_path}: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Change file permissions recursively.")
    parser.add_argument(
        'directory',
        nargs='?',
        default='F:/',
        help='Directory path to change file permissions. Default is F:/'
    )
    args = parser.parse_args()
    
    # Run the permission change function
    change_permissions_if_read_only(args.directory)

if __name__ == "__main__":
    main()
