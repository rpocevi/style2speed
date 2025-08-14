# style2speed/utils.py

def download_file_if_colab(filepath):
    """
    Triggers a file download if running in Google Colab.

    Parameters:
        filepath (str): Path to the file to be downloaded.
    """
    try:
        from google.colab import files
        files.download(filepath)
        print(f'Triggered download: {filepath}')
    except ImportError:
        print(f'Download skipped (not running in Colab): {filepath}')