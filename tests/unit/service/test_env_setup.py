import unittest
from unittest.mock import patch, mock_open
import subprocess
import sys
from service.env_setup import install_package, install_packages_from_file

class TestEnvSetup(unittest.TestCase):

    @patch('subprocess.check_call')
    def test_install_package_success(self, mock_check_call) -> None:
        mock_check_call.return_value = 0
        result: bool = install_package('requests')
        self.assertTrue(result)
        mock_check_call.assert_called_once_with([sys.executable, '-m', 'pip', 'install', 'requests', "--no-cache-dir", "--no-warn-script-location"])
    
    @patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'pip'))
    def test_install_package_failure(self, mock_check_call) -> None:
        result: bool = install_package('non_existent_package')
        self.assertFalse(result)
        mock_check_call.assert_called_once_with([sys.executable, '-m', 'pip', 'install', 'non_existent_package', "--no-cache-dir", "--no-warn-script-location"])

    @patch('builtins.open', new_callable=mock_open, read_data='requests\n')
    @patch('subprocess.check_call')
    def test_install_packages_from_file(self, mock_check_call, mock_file) -> None:
        mock_check_call.return_value = 0
        install_packages_from_file('requirements.txt')
        mock_file.assert_called_once_with('requirements.txt', 'r')
        mock_check_call.assert_called_with([sys.executable, '-m', 'pip', 'install', 'requests', "--no-cache-dir", "--no-warn-script-location"])

    @patch('builtins.open', new_callable=mock_open, read_data='non_existent_package\n')
    @patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'pip'))
    def test_install_packages_from_file_failure(self, mock_check_call, mock_file) -> None:
        install_packages_from_file('requirements.txt')
        mock_file.assert_called_once_with('requirements.txt', 'r')
        mock_check_call.assert_called_with([sys.executable, '-m', 'pip', 'install', 'non_existent_package', "--no-cache-dir", "--no-warn-script-location"])

if __name__ == '__main__':
    unittest.main()
