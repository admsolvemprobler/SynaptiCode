#!/usr/bin/env python3
"""
Tests for the Change Propagator
"""

import unittest
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.propagator import ChangePropagator

class TestChangePropagator(unittest.TestCase):
    """Test cases for the ChangePropagator class."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a simple mock network
        self.network = {
            "file1.py": ["file2.py", "file3.py"],
            "file2.py": ["file4.py"],
            "file3.py": [],
            "file4.py": ["file1.py"]
        }
        self.propagator = ChangePropagator(self.network)
        
    def test_detect_change(self):
        """Test the detect_change method."""
        # This is a placeholder test
        result = self.propagator.detect_change("file1.py", "old content")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["file"], "file1.py")
        
    def test_propagate(self):
        """Test the propagate method."""
        # This is a placeholder test
        change = {"type": "modification", "file": "file1.py"}
        result = self.propagator.propagate(change)
        self.assertIsInstance(result, list)
        
if __name__ == '__main__':
    unittest.main()
