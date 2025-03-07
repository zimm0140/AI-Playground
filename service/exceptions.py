"""
Custom Exception Classes
-----------------------
This module defines custom exception classes used throughout the application
for specific error scenarios related to downloading operations.
"""

class DownloadException(Exception):
    """
    Exception raised when a file download operation fails.
    
    This exception is typically raised after multiple retry attempts
    have failed when downloading a file from a URL.
    
    Attributes:
        url: The URL of the resource that failed to download.
    """
    url: str

    def __init__(self, url: str):
        """
        Initialize the DownloadException with the URL that failed.
        
        Args:
            url: The URL of the resource that failed to download.
        """
        super().__init__(f"download {url} failed")
