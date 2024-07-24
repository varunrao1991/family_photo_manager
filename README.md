# family_photo_manager
Manage all your personal photos

Step 1. Create a virtual environment
            python3 -m venv myenv
Step 2. Install dependencies
            pip3 install -r requirements.txt
Step 3. Activate environment
            source myenv/bin/activate
Step 4. Create data folder in current directory and fill with the images
Step 5. Run database creation script
            python3 image_similarity.py ./data
Step 6. Run flask app
            python3 app.py