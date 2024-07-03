import os
import shutil
from datetime import datetime


def save_best_model(current_best_path, latest_dir, archive_dir):
    """Save the latest best trained model to models/checkpoints/latest and move previous to models/checkpoints/archive

    Args:
        current_best_path (str): path to current best model
        latest_dir (str): path to latest directory
        archive_dir (str): path to archive directory
    """
    # Create directories if they don't exist
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    # Archive the current best model if it exists
    if os.path.exists(os.path.join(latest_dir, 'best.pt')):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(archive_dir, f'{timestamp}_best.pt')
        shutil.move(os.path.join(latest_dir, 'best.pt'), archive_path)
    
    # Save the new best model
    shutil.copy(current_best_path, os.path.join(latest_dir, 'best.pt'))


if __name__ == "__main__":
    current_best_path = 'output/experiment_9/weights/best.pt'  # Update with the correct path
    latest_dir = 'models/checkpoints/latest'
    archive_dir = 'models/checkpoints/archive'
    save_best_model(current_best_path, latest_dir, archive_dir)
